# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

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


from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer, MultiTokenPredictionBlock, get_mtp_layer_offset

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
