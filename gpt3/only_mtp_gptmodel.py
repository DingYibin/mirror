# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, Literal, Optional
import os
import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MultiTokenPredictionBlock,
    roll_tensor,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params
    
    
from megatron.core.models.gpt import GPTModel

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

class _ParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mtp_logits, target_prob):
        """Vocab parallel cross entropy forward function."""


        mtp_logits = mtp_logits.float()
        logits_max = torch.max(mtp_logits, dim=-1)[0]

        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )


        mtp_logits -= logits_max.unsqueeze(dim=-1)

        sum_exp_logits = torch.exp(mtp_logits).sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        mtp_logits -= torch.log(sum_exp_logits.unsqueeze(dim=-1))

        loss = torch.sum(-target_prob * mtp_logits, 2)

        torch.distributed.all_reduce(
            loss,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(mtp_logits, target_prob)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Vocab parallel cross entropy backward function."""

        # Retreive tensors from the forward path.
        mtp_logits, target_prob = ctx.saved_tensors
        grad_input = torch.exp(mtp_logits) - target_prob
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def parallel_cross_entropy(mtp_logits, target_prob):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        mtp_logits: logits split across tensor parallel ranks
            dimension is [sequence_length, batch_size, vocab_size/num_parallel_ranks]

        target_prob: correct vocab ids of dimseion [sequence_length, batch_size, vocab_size/num_parallel_ranks]

    """
    return _ParallelCrossEntropy.apply(mtp_logits, target_prob)


MTP_EH_PROJ_MODE = int(os.environ.get("MTP_EH_PROJ_MODE", "0"))
NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER = int(os.environ.get("NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER", "1"))

class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers + 1, device=torch.cuda.current_device())
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Reduce mtp losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        mtp_losses = tracker["values"] * loss_scale
        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            name = f"mtp_{i} loss" if i > 0 else "ideal loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                if name in total_loss_dict:
                    total_loss_dict[name] += loss
                else:
                    total_loss_dict[name] = loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()

class OnlyMTPGPTModel(GPTModel):

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        )

        with torch.no_grad():
            # Run decoder.
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **(extra_block_kwargs or {}),
            )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        in_inference_mode = inference_context is not None and not self.training
        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if mtp_in_postprocess:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if self.mtp_process:
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]

        sequence_parallel_override = False
        if in_inference_mode and inference_context.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    # TODO(ksanthanam): Make the equivalent change in the `MambaModel` code after
                    # merging in !3722.
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.model_comm_pgs.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
                # state ([B, H]) → unsqueeze back to [1, B, H]
                # (so that the output layer, which expects S×B×H, receives only the final token)
                hidden_states = inference_context.last_token_logits(
                    hidden_states.squeeze(1).unsqueeze(0)
                ).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        
        if self.mtp_process:
            target_logits = logits.float().detach()
            target_logits_max = torch.max(target_logits, dim=-1)[0]

            torch.distributed.all_reduce(
                target_logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
            )
            target_logits -= target_logits_max.unsqueeze(dim=-1)
            target_prob = target_logits.clone()
            torch.exp(target_logits, out=target_prob)
            sum_target_prob = target_prob.sum(dim=-1)
            torch.distributed.all_reduce(
                sum_target_prob,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )
            target_prob.div_(sum_target_prob.unsqueeze(dim=-1))
            target_prob = target_prob.detach()
            target_logits = (target_logits - sum_target_prob.log().unsqueeze(dim=-1)).detach()
            ideal_loss = -(target_logits * target_prob).sum(dim=-1)
            torch.distributed.all_reduce(
                ideal_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )


            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)
            # loss_mask[:, :loss_mask.shape[1] // 2].fill_(0)

            mtp_num_layers = self.config.mtp_num_layers
            mtp_num_layers_step = 1
            if MTP_EH_PROJ_MODE == 10 and NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER > 1:
                mtp_num_layers = self.config.mtp_num_layers // NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER
                mtp_num_layers_step = NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER

            if self.training:
                MTPLossLoggingHelper.save_loss_to_tracker(
                    torch.sum(ideal_loss) / loss_mask.sum(),
                    0,
                    mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    ),
                )

            for mtp_layer_number in range(mtp_num_layers):
                hidden_states_index = (mtp_layer_number + 1) * mtp_num_layers_step
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[hidden_states_index],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                target_prob, _ = roll_tensor(target_prob, shifts=-1, dims=0, cp_group=self.cp_group)
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
                )


                mtp_loss = parallel_cross_entropy(mtp_logits, target_prob)
                if self.training:
                    # TODO(shifangx): remove the use of parallel_state here
                    # after moving loss logging to loss_func in pretrain_gpt.py
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number + 1,
                        mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / mtp_num_layers
                if self.config.calculate_per_token_loss:
                    logits = MTPLossAutoScaler.apply(
                        logits, mtp_loss_scale * mtp_loss
                    )
                else:
                    logits = MTPLossAutoScaler.apply(
                        logits, mtp_loss_scale * mtp_loss / num_tokens
                    )

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss