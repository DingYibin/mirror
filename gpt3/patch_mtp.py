# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import os
import torch
from torch import Tensor

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    make_viewless_tensor,
)

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionBlock,
    get_mtp_layer_offset,
    MultiTokenPredictionLayerSubmodules,
    ModelCommProcessGroups,
    SUPPORTED_ATTN_MASK,
    MegatronModule,
)
def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return torch.nn.functional.silu(x[0]) * x[1]
            
# class AdjustedMultiTokenPredictionLayer(MultiTokenPredictionLayer):

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

    # 0: original
    # 1: move eh_proj to last
    # 2: move_eh_proj to last and add to embeddings
    # 3: move_eh_proj to last and add to hidden states
    # 4: remove eh_proj
    self.eh_proj_mode = int(os.environ.get("MTP_EH_PROJ_MODE", "1"))
    if self.eh_proj_mode < 1 or self.eh_proj_mode > 4:
        self.eh_proj_mode = 1
    print(f"EH_PROJ_RESNET_MODE = {self.eh_proj_mode}\n", end="")

    if self.eh_proj_mode != 4:

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
        self.eh_proj = build_module(
            self.submodules.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        self.activation_func = swiglu
    self.transformer_layer = build_module(
        self.submodules.transformer_layer, config=self.config, vp_stage=vp_stage
    )

    self.final_layernorm = build_module(
        self.submodules.layer_norm,
        config=self.config,
        hidden_size=self.config.hidden_size,
        eps=self.config.layernorm_epsilon,
    )
    self.offload_context = nullcontext()

def MultiTokenPredictionLayer_concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
    """
    Concatenate the tokens before sending to transformer layer.
    """
    if self.eh_proj_mode == 4:
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
    if self.eh_proj_mode == 1:
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

MultiTokenPredictionLayer.__init__ = MultiTokenPredictionLayer__init__
MultiTokenPredictionLayer._concat_embeddings = MultiTokenPredictionLayer_concat_embeddings
MultiTokenPredictionLayer._proj_and_transformer_layer = MultiTokenPredictionLayer_proj_and_transformer_layer



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
        hidden_states, out_hidden_states = hidden_states_re
        # append the output hidden states of the current mtp layer
        # to the hidden_states_list
        hidden_states_list.append(out_hidden_states)

    # concat the hidden states of all mtp layers
    hidden_states = torch.cat(hidden_states_list, dim=0)
    return hidden_states

MultiTokenPredictionBlock.forward = MultiTokenPredictionBlock_forward

print("Patch to move eh_proj has been applied!\n", end="", flush=True)