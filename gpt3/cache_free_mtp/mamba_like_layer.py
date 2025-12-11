# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params


@dataclass
class MambaLikeLayerSubmodules:
    """
    Configuration class for specifying the submodules of a Mamba layer.

    This class defines the structure and default implementations for various
    components of a Mamba layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        mamba_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
    """

    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaLikeLayer(MegatronModule):
    """
    A single Mamba layer.

    Mamba layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32=False,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        """Initialize Mamba Layer."""
        super().__init__(config)
        assert model_comm_pgs is not None, "model_comm_pgs must be provided for MambaLayer"

        if config.enable_cuda_graph and config.cuda_graph_scope != "full_iteration":
            self.cudagraph_manager = CudaGraphManager(config)

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=layer_number,
            model_comm_pgs=model_comm_pgs,
        )
        self.norm = build_module(submodules.norm, self.config, self.config.hidden_size)
        self.mamba_bda = build_module(submodules.mamba_bda)
        
        
        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        additional_mlp_kwargs = {}
        # import here to avoid circular import
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects model_comm_pgs to be passed in.
        # We can change MLP to accept model_comm_pgs but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["model_comm_pgs"] = model_comm_pgs
            elif submodules.mlp.module == MLP:
                assert hasattr(
                    model_comm_pgs, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = model_comm_pgs.tp
            elif TEFusedMLP is not None and submodules.mlp.module == TEFusedMLP:
                assert hasattr(
                    model_comm_pgs, 'tp'
                ), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = model_comm_pgs.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(submodules.mlp)}. Using default kwargs.",
                )
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if (
                    not isinstance(self.input_layernorm, IdentityOp)
                    and not self.config.external_cuda_graph
                ):
                    self.recompute_input_layernorm = True
                    if self.config.fp8:
                        self.self_attention.set_for_recompute_input_layernorm()
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8:
                        if isinstance(self.mlp, MoELayer):
                            self.mlp.set_for_recompute_pre_mlp_layernorm()
                        else:
                            from megatron.core.extensions.transformer_engine import (
                                set_save_original_input,
                            )

                            set_save_original_input(self.mlp.linear_fc1)
            if "mlp" in self.config.recompute_modules:
                if not isinstance(self.mlp, MoELayer):
                    self.recompute_mlp = True

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad


    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        hidden_states, context = self._forward_attention(*args, **kwargs)
        output = self._forward_mlp(hidden_states, kwargs.get("inference_context", None))
        return output, context

    def _forward_attention(
        self,
        hidden_states: Tensor,
        ssm_state: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform a forward pass through the Mamba layer.

        This method implements the core computation of a Mamba layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = self.norm(hidden_states)

        hidden_states, hidden_states_bias, ssm_state = self.mixer(
            hidden_states, ssm_state, inference_context=inference_context)

        return hidden_states, ssm_state


    def _forward_mlp(self, hidden_states, inference_context=None):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
        )

        if self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    pre_mlp_layernorm_output,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
            outputs = [self.mlp(chunk) for chunk in chunks]

            # Aggregate chunk outputs
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)

        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        nvtx_range_pop(suffix="mlp")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="mlp_bda")

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate the inference cache."""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def __call__(self, *args, **kwargs):

        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return self.cudagraph_manager(self, args, kwargs)
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs['inference_context'].is_decode_only()
        ):
            return self.cudagraph_manager(self, args, kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the mamba layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the mamba layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
