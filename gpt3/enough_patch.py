# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import os
# TODO move into config
NUM_LAYERS_PART_ONE = int(os.environ.get("NUM_LAYERS_PART_ONE", "0"))
NUM_LAYERS_PART_TWO = int(os.environ.get("NUM_LAYERS_PART_TWO", "0"))
NUM_PREDICTION_TOKENS = int(os.environ.get("NUM_PREDICTION_TOKENS", "1"))


from collections import OrderedDict
import copy

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Dict, Literal, Optional

import torch
from torch import Tensor

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.fp8_utils import get_fp8_context
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
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    is_torch_min_version,
    make_viewless_tensor,
    WrappedTensor,
)

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base


from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionBlock,
    get_mtp_layer_offset,
    MultiTokenPredictionLayerSubmodules,
    ModelCommProcessGroups,
    SUPPORTED_ATTN_MASK,
    MegatronModule,
    roll_tensor,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return torch.nn.functional.silu(x[0]) * x[1]

class IdentityOpTwoOutput(torch.nn.Module):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x, x

class EnoughLayer(MultiTokenPredictionLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        assert layer_number < NUM_LAYERS_PART_ONE + NUM_LAYERS_PART_TWO
        MegatronModule.__init__(self, config=config)
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number
        self.vp_stage = vp_stage
        self.cp_group = model_comm_pgs.cp

        self_attention_spec = self.submodules.transformer_layer.submodules.self_attention
        attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            f"Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        if self.layer_number == NUM_LAYERS_PART_ONE:
            self.enorm = build_module(
                self.submodules.enorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        if self.layer_number == 0 or self.layer_number == NUM_LAYERS_PART_ONE:
            self.hnorm = build_module(
                self.submodules.hnorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        if self.layer_number == 0 or self.layer_number == NUM_LAYERS_PART_ONE:

            # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
            # of the i-th tocken's hidden states and the (i + K)-th tocken's decoder input,
            # so the input's shape is [s, b, 2*h].
            # The output will be send to the following transformer layer,
            # so the output's shape should be [s, b, h].
            if self.layer_number == 0:
                eh_proj_input_hidden_size = self.config.hidden_size
            elif self.layer_number == NUM_LAYERS_PART_ONE:
                eh_proj_input_hidden_size = self.config.hidden_size * 2

            if self.layer_number == 0:
                eh_proj_output_hidden_size = self.config.hidden_size * (NUM_PREDICTION_TOKENS + 1)
            elif self.layer_number == NUM_LAYERS_PART_ONE:
                eh_proj_output_hidden_size = self.config.hidden_size

            self.eh_proj = build_module(
                self.submodules.eh_proj,
                eh_proj_input_hidden_size,
                eh_proj_output_hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        self.transformer_layer = build_module(
            self.submodules.transformer_layer, config=self.config, vp_stage=vp_stage
        )
        if self.layer_number + 1 == NUM_LAYERS_PART_ONE + NUM_LAYERS_PART_TWO:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = torch.nn.Identity()
        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
        """
        decoder_input = None
        if self.layer_number == NUM_LAYERS_PART_ONE:
            # Calc logits for the current Multi-Token Prediction (MTP) layers.
            input_ids_list = [input_ids]
            position_ids_list = [position_ids]
            for _ in range(NUM_PREDICTION_TOKENS - 1):
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
                input_ids_list.append(input_ids)
                position_ids, _ = roll_tensor(position_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
                position_ids_list.append(position_ids)
            input_ids = torch.stack(input_ids_list, dim=-1)
            position_ids = torch.stack(position_ids_list, dim=-1)
            # embedding
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return input_ids, position_ids, decoder_input, hidden_states



    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):

        if self.layer_number == 0 or self.layer_number == NUM_LAYERS_PART_ONE:
            """
            Concatenate the tokens before sending to transformer layer.
            """
            s, b, d = hidden_states.shape
            if self.layer_number == NUM_LAYERS_PART_ONE:
                decoder_input = self.enorm(decoder_input)
                decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
            hidden_states = self.hnorm(hidden_states)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
            # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
            # and the (i + K)-th tocken's embedding, and combine them with linear projection.
            if self.layer_number == NUM_LAYERS_PART_ONE:
                hidden_states = torch.cat((decoder_input, hidden_states), -1)
            hidden_states, _ = self.eh_proj(hidden_states)
            # For tensor parallel we need to gather the tensor across the model-parallel
            # ranks after the linear projection. This used to call
            # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
            # the gradient in backward pass and was therefore incorrect in this context.
            # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
            hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
            # For sequence parallel, scatter after linear_fc and before transformer layer.
            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(hidden_states)
            if self.layer_number == 0:
                hidden_states = hidden_states.view(s * b, -1, d)
            return hidden_states
        else:
            return hidden_states

class EnoughBlock(MultiTokenPredictionBlock):


    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        extra_block_kwargs: dict = None,
        embedding=None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """


        for layer_number in range(len(self.layers)):
            (hidden_states, input_ids, position_ids) = self.layers[layer_number](
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
                embedding=embedding,
                **(extra_block_kwargs or {}),
            )

        return hidden_states

class EnoughModel(GPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        LanguageModule.__init__(self, config=config, model_comm_pgs=model_comm_pgs)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage

        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 4 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if hasattr(self.config, 'rotary_base'):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.model_comm_pgs.tp,
            )

        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.model_comm_pgs.cp,
            )

        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = MultimodalRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )
            self.mrope_section = self.config.mrope_section
            assert (
                self.mrope_section is not None
            ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            model_comm_pgs=self.model_comm_pgs,
            vp_stage=vp_stage,
        )

        if self.mtp_process:
            self.mtp = EnoughBlock(
                config=self.config, spec=self.mtp_block_spec, vp_stage=vp_stage
            )

        # Output
        if self.post_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.model_comm_pgs.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )
        for name, module in self.named_modules():
            if hasattr(module, 'finish_init'):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

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
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = torch.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_num_layers):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group)
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss
                if self.training:
                    # TODO(shifangx): remove the use of parallel_state here
                    # after moving loss logging to loss_func in pretrain_gpt.py
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        torch.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )
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
