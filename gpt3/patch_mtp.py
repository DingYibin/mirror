# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import os
"""
0: original
1: move eh_proj to last, add swiglu
2: move_eh_proj to last and add to embeddings, add swiglu
3: move_eh_proj to last and add to hidden states, add swiglu
4: remove eh_proj
5: just move eh_proj to last
6: remove embed input, keep eh_proj
7: keep eh_proj, remove hidden states
8: remove self attn
"""
import copy
MTP_EH_PROJ_MODE = int(os.environ.get("MTP_EH_PROJ_MODE", "0"))
NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER = int(os.environ.get("NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER", "1"))

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    is_torch_min_version,
    make_viewless_tensor,
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


def MultiTokenPredictionLayer__init__(
    self,
    config: TransformerConfig,
    submodules: MultiTokenPredictionLayerSubmodules,
    layer_number: int = 1,
    vp_stage: Optional[int] = None,
    model_comm_pgs: ModelCommProcessGroups = None,
):
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

    self.eh_proj_mode = MTP_EH_PROJ_MODE
    self.num_transformer_block_one_mtp_layer = NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER
    assert self.eh_proj_mode != 10 or self.num_transformer_block_one_mtp_layer > 0, f"MTP_EH_PROJ_MODE = {self.eh_proj_mode}, NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER should be geater than 0, which is {self.num_transformer_block_one_mtp_layer}"


    if self.eh_proj_mode in [1, 2, 3, 5, 6, 7, 8, 9] or (self.eh_proj_mode == 10 and (self.layer_number - 1) % self.num_transformer_block_one_mtp_layer == 0):

        self.enorm = build_module(
            self.submodules.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th tocken's hidden states and the (i + K)-th tocken's decoder input,
        # so the input's shape is [s, b, 2*h].
        # The output will be send to the following transformer layer,
        # so the output's shape should be [s, b, h].
        if self.eh_proj_mode in [1, 2, 3, 5, 8, 9, 10]:
            eh_proj_input_hidden_size = self.config.hidden_size * 2
        elif self.eh_proj_mode in [6, 7]:
            eh_proj_input_hidden_size = self.config.hidden_size

        if self.eh_proj_mode in [1, 2, 3]:
            eh_proj_output_hidden_size = self.config.hidden_size * 2
        elif self.eh_proj_mode in [5, 6, 7, 8, 9, 10]:
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
        if self.eh_proj_mode in [1, 2, 3]:
            self.activation_func = swiglu
        elif self.eh_proj_mode in [5, 6, 7, 8, 9, 10]:
            self.activation_func = torch.nn.Identity()
    transformer_layer_modules = copy.deepcopy(self.submodules.transformer_layer)
    if self.eh_proj_mode == 8 or (self.eh_proj_mode == 9 and layer_number > 1):
        transformer_layer_modules.submodules.self_attention = IdentityOpTwoOutput
    self.transformer_layer = build_module(
        transformer_layer_modules, config=self.config, vp_stage=vp_stage
    )
    if self.eh_proj_mode == 10 and self.layer_number % self.num_transformer_block_one_mtp_layer != 0:
        self.final_layernorm = torch.nn.Identity()
    else:
        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
    self.offload_context = nullcontext()

def MultiTokenPredictionLayer_get_embeddings(
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
    if self.eh_proj_mode != 10  or (self.eh_proj_mode == 10 and (self.layer_number - 1) % self.num_transformer_block_one_mtp_layer == 0):
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
        position_ids, _ = roll_tensor(position_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
    # embedding
    decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    return input_ids, position_ids, decoder_input, hidden_states

def MultiTokenPredictionLayer_concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
    """
    Concatenate the tokens before sending to transformer layer.
    """
    # just remove eh_proj
    if self.eh_proj_mode == 4 or (self.eh_proj_mode == 10 and (self.layer_number - 1) % self.num_transformer_block_one_mtp_layer != 0):
        return hidden_states
    if self.eh_proj_mode in [6, 7, 10]:
        if self.eh_proj_mode == 6:
            # remove embed
            hidden_states = self.hnorm(hidden_states)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        elif self.eh_proj_mode == 7:
            hidden_states = self.enorm(decoder_input)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        elif self.eh_proj_mode == 10:
            decoder_input = self.enorm(decoder_input)
            decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
            hidden_states = self.hnorm(hidden_states)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
            # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
            # and the (i + K)-th tocken's embedding, and combine them with linear projection.
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
        return hidden_states
    hidden_states_input = hidden_states
    embeddings = self.enorm(decoder_input)
    embeddings = make_viewless_tensor(inp=embeddings, requires_grad=True, keep_graph=True)
    
    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
    # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
    # and the (i + K)-th tocken's embedding, and combine them with linear projection.
    hidden_states = torch.cat((embeddings, hidden_states), -1)
    hidden_states, _ = self.eh_proj(hidden_states)
    # For tensor parallel we need to gather the tensor across the model-parallel
    # ranks after the linear projection. This used to call
    # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
    # the gradient in backward pass and was therefore incorrect in this context.
    # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
    hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
    hidden_states = self.activation_func(hidden_states)
    if self.eh_proj_mode in [1, 5]:
        hidden_states = self.hnorm(hidden_states)
    elif self.eh_proj_mode == 2:
        hidden_states = self.hnorm(hidden_states + decoder_input)
    elif self.eh_proj_mode == 3:
        hidden_states = self.hnorm(hidden_states + hidden_states_input)
        
    # For sequence parallel, scatter after linear_fc and before transformer layer.
    if self.sequence_parallel:
        hidden_states = scatter_to_sequence_parallel_region(hidden_states)
    return hidden_states


def MultiTokenPredictionLayer_proj_and_transformer_layer(
    self,
    hidden_states: Tensor,
    decoder_input: Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    rotary_pos_cos: Optional[torch.Tensor] = None,
    rotary_pos_sin: Optional[torch.Tensor] = None,
    attention_bias: Optional[torch.Tensor] = None,
    inference_params: Optional[InferenceParams] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Concatenates embeddings with hidden states and then applies transformer layer forward.
    """
    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    # Unlike transformer_block.py which needs to support mixed-precision in
    # different layers,currently MTP only use global fp8 context.
    if self.config.fp8:
        fp8_context = get_fp8_context(self.config)
        transformer_layer_fp8_context = get_fp8_context(self.config)
    else:
        fp8_context = nullcontext()
        transformer_layer_fp8_context = nullcontext()

    with rng_context:
        

        # Use a separate fp8 context for the transformer layer. This is to ensure that when the
        # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
        # True so that the fp8 weight caching can be triggered correctly.
        with transformer_layer_fp8_context:
            hidden_states, _ = self.transformer_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        with fp8_context:
            out_hidden_states = self._concat_embeddings(hidden_states, decoder_input)
    hidden_states = self._postprocess(hidden_states)

    return hidden_states, out_hidden_states

def MultiTokenPredictionBlock_forward(
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
    # get hidden states from previous mtp stages
    offset = get_mtp_layer_offset(self.config)
    hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
    hidden_states = hidden_states_list[offset]
    # print("MultiTokenPredictionBlock_forward")
    for layer_number in range(len(self.layers)):
        (hidden_states_re, input_ids, position_ids) = self.layers[layer_number](
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
        if isinstance(hidden_states_re, tuple):
            hidden_states, out_hidden_states = hidden_states_re
        else:
            hidden_states, out_hidden_states = hidden_states_re, hidden_states_re # MODE 10
        # append the output hidden states of the current mtp layer
        # to the hidden_states_list
        hidden_states_list.append(out_hidden_states)

    # concat the hidden states of all mtp layers
    hidden_states = torch.cat(hidden_states_list, dim=0)
    return hidden_states


patched_info = f"{MTP_EH_PROJ_MODE = }\n{NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER = }\n"
if MTP_EH_PROJ_MODE in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    MultiTokenPredictionLayer.__init__ = MultiTokenPredictionLayer__init__
    patched_info += "MultiTokenPredictionLayer.__init__ is replaced.\n"
if MTP_EH_PROJ_MODE in [                           10]:
    MultiTokenPredictionLayer._get_embeddings = MultiTokenPredictionLayer_get_embeddings
    patched_info += "MultiTokenPredictionLayer._get_embeddings is replaced.\n"
# if MTP_EH_PROJ_MODE not in [8, 9]:
if MTP_EH_PROJ_MODE in [1, 2, 3, 4, 5, 6, 7,       10]:
    MultiTokenPredictionLayer._concat_embeddings = MultiTokenPredictionLayer_concat_embeddings
    patched_info += "MultiTokenPredictionLayer._concat_embeddings is replaced.\n"
# if MTP_EH_PROJ_MODE not in [6, 7, 8, 9]:
if MTP_EH_PROJ_MODE in [1, 2, 3, 4, 5,               ]:
    MultiTokenPredictionLayer._proj_and_transformer_layer = MultiTokenPredictionLayer_proj_and_transformer_layer
    patched_info += "MultiTokenPredictionLayer._proj_and_transformer_layer is replaced.\n"
    MultiTokenPredictionBlock.forward = MultiTokenPredictionBlock_forward
    patched_info += "MultiTokenPredictionBlock.forward is replaced.\n"

patched_info += "Patch to move eh_proj has been applied!\n"
print(patched_info, end="", flush=True)