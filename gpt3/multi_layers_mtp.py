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
)


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
        self.config.hidden_size,
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

    self.final_layernorm = build_module(
        self.submodules.layer_norm,
        config=self.config,
        hidden_size=self.config.hidden_size,
        eps=self.config.layernorm_epsilon,
    )
    self.offload_context = nullcontext()