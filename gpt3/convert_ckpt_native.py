import os
import torch
import re
from safetensors.torch import save_file
import json



def convert_weights(
        source_dir,
        target_dir,
        tp_size,
        pp_size,
        ep_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        num_experts=None,
        consider_multi_inner=False,
):
    
    weights = [{} for _ in range(tp_size)]

    keys = list()
    num_experts_per_ep = 0 if num_experts is None else num_experts // ep_size
    expert_pattern = re.compile(r'.linear_fc(1|2).weight(\d+)$')
    for i in range(tp_size):
        for j in range(pp_size):
            for k in range(ep_size):
                dir_name = f'mp_rank_{i:02d}'
                if pp_size > 1:
                    dir_name += f'_{j:03d}'
                if ep_size > 1:
                    dir_name += f'_{k:03d}'
                filepath = os.path.join(source_dir, dir_name, "model_optim_rng.pt")
                print(filepath)
                if weights[i] is None:
                    weights[i] = {}
                one_weight = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)['model']
                for key, value in one_weight.items():
                    if key.startswith('mtp') and "_extra_state" not in key:
                        if 'experts' in key:
                            match = expert_pattern.search(key)
                            expert_id = k * num_experts_per_ep + int(match.group(2))
                            new_key = f"{key[:match.start(0)]}.{expert_id}.linear_fc{match.group(1)}.weight"
                            weights[i][new_key] = value
                            keys.append(new_key)
                            # print(new_key)
                        elif key not in weights[i]:
                            weights[i][key] = value
                            keys.append(key)
    merged_weight = dict()
    for key in keys:
        if "norm" in key or "mlp.router.weight" in key:
            for i in range(tp_size):
                val = weights[i].get(key)
                if val is not None:
                    merged_weight[key] = val
                    break
        elif key.endswith(("eh_proj.weight",)):
            # column parallel
            vals = [weights[i][key] for i in range(tp_size)]
            val = torch.cat(vals, dim=0)
            merged_weight[key] = val
        elif key.endswith(("linear_fc1.weight",)):
            vals = [weights[i][key].view(2, -1, hidden_size) for i in range(tp_size)]
            val = torch.cat(vals, dim=1).view(-1, hidden_size)
            merged_weight[key] = val
        elif key.endswith(("self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias")):
            dim = weights[0][key].shape[0] * tp_size
            head_dim = dim // (num_attention_heads + num_key_value_heads * 2)
            num_per_group = num_attention_heads // num_key_value_heads
            num_group_per_tp = num_key_value_heads // tp_size
            if key.endswith("weight"):
                vals = [weights[i][key].view(num_group_per_tp, -1, head_dim, weights[i][key].shape[-1]) for i in range(tp_size)]
            elif key.endswith("bias"):
                vals = [weights[i][key].view(num_group_per_tp, -1, head_dim) for i in range(tp_size)]
            # print(", ".join([f"{item.shape}" for item in vals]))
            val = torch.cat(vals, dim=0)
            merged_weight[key] = val
        elif key.endswith(("linear_fc2.weight", "self_attention.linear_proj.weight")):
            # row parallel
            vals = [weights[i][key] for i in range(tp_size)]
            val = torch.cat(vals, dim=1)
            merged_weight[key] = val
        else:
            out_str = f"{key} : "
            for i in range(tp_size):
                val = weights[i].get(key, torch.tensor([]))
                out_str += f" {val.shape}"
            print("unpropossed: " + out_str)

    

    converted_weight = dict()
    mtp_names = set()
    for key, val in merged_weight.items():
        key = key.replace("mtp.layers", "model.mtp")
        mtp_name = ".".join(key.split(".")[:3])
        mtp_names.add(mtp_name)
        
        key = key.replace('.transformer_layer.', ".layers.0." if consider_multi_inner else '.')

        key = key.replace('self_attention', 'self_attn')

        key = key.replace('self_attn.linear_qkv.layer_norm_weight', 'input_layernorm.weight')

        key = key.replace('pre_mlp_layernorm.weight', 'post_attention_layernorm.weight')
        key = key.replace('mlp.linear_fc1.layer_norm_weight', 'post_attention_layernorm.weight')

        key = key.replace('self_attn.linear_proj.weight', 'self_attn.o_proj.weight')

        key = key.replace('linear_fc2.weight', 'down_proj.weight')

        key = key.replace('mlp.router.weight', 'mlp.gate.weight')

        if key.endswith(('linear_qkv.weight', 'linear_qkv.bias')):
            dim = val.shape[0]
            head_dim = dim // (num_attention_heads + num_key_value_heads * 2)
            num_per_group = num_attention_heads // num_key_value_heads
            q_proj, k_proj, v_proj = torch.split(val, [num_per_group, 1, 1], dim=1)
            # print(f"{q_proj.shape = }, {k_proj.shape = }, {v_proj.shape = }")
            if key.endswith('weight'):
                q_proj = q_proj.reshape(-1, q_proj.shape[-1])
                k_proj = k_proj.reshape(-1, k_proj.shape[-1])
                v_proj = v_proj.reshape(-1, v_proj.shape[-1])
            elif key.endswith('bias'):
                q_proj = q_proj.reshape(-1)
                k_proj = k_proj.reshape(-1)
                v_proj = v_proj.reshape(-1)

            q_key = key.replace('linear_qkv', 'q_proj')
            converted_weight[q_key] = q_proj

            k_key = key.replace('linear_qkv', 'k_proj')
            converted_weight[k_key] = k_proj

            v_key = key.replace('linear_qkv', 'v_proj')
            converted_weight[v_key] = v_proj
        elif key.endswith('linear_fc1.weight'):
            gate_proj, up_proj = torch.chunk(val, 2, 0)

            gate_key = key.replace('linear_fc1', 'gate_proj')
            converted_weight[gate_key] = gate_proj

            up_key = key.replace('linear_fc1', 'up_proj')
            converted_weight[up_key] = up_proj
        else:
            converted_weight[key] = val

    del merged_weight
    mtp_names = list(mtp_names)
    num_transformer_layers = len(mtp_names)
    print(mtp_names)

    weights_per_layer = [dict() for _ in range(num_transformer_layers)]
    for key, val in converted_weight.items():
        layer_id = int(key.split(".")[2])
        weights_per_layer[layer_id][key] = val

    if consider_multi_inner:
        split_weights = list()
        num_inner_layers = 0
        num_mtp = 0
        for i in range(num_transformer_layers):
            if num_mtp >= len(split_weights):
                split_weights.append(dict())
            for key, val in weights_per_layer[i].items():
                key = key.replace(f"model.mtp.{i}.", f"model.mtp.{num_mtp}.")
                key = key.replace(f"layers.0.", f"layers.{num_inner_layers}.")
                split_weights[num_mtp][key] = val
            
            num_inner_layers += 1

            final_layernorm_name = f"model.mtp.{i}.final_layernorm.weight"
            if final_layernorm_name in weights_per_layer[i]:
                num_inner_layers = 0
                num_mtp += 1
    else:
        split_weights = weights_per_layer
    
    os.makedirs(target_dir, exist_ok=True)
    total_size = 0
    weight_map = dict()
    for i, layer_weights in enumerate(split_weights):
        filename = f"model-{i + 1:05d}-of-{len(split_weights):05d}.safetensors"
        filepath = os.path.join(target_dir, filename)
        for key, val in layer_weights.items():
            print(key, val.shape, filename)
            total_size += val.numel() * val.element_size()
            weight_map[key] = filename

        save_file(layer_weights, filepath)

    index_info = {
        'metadata' : {'total_size': total_size},
        'weight_map' : weight_map,
    }
    with open(os.path.join(target_dir, "model.safetensors.index.json"), 'w', encoding='utf-8') as f:
        json.dump(index_info, f, ensure_ascii=False, indent=4)



# convert_weights(
#     source_dir="/workspace-dyb/experiments/ckpt/qwen-30b-a3b-thinking-tp1-ep8/torch/iter_0016384/",
#     target_dir="/workspace-dyb/hw-ding/qwen3-30b-a3b-thinking-mtp-7",
#     tp_size=1,
#     pp_size=1,
#     ep_size=8,
#     hidden_size=2048,
#     num_attention_heads=32,
#     num_key_value_heads=4,
#     num_experts=128,
# )


# convert_weights(
#     source_dir="/workspace-dyb/experiments/ckpt/qwq-mode-10-2-1220/torch/torch/iter_0016384/",
#     target_dir="/workspace-dyb/hw-ding/qwq-32b-mtp-7-2",
#     tp_size=4,
#     pp_size=1,
#     ep_size=1,
#     hidden_size=5120,
#     num_attention_heads=40,
#     num_key_value_heads=8,
#     consider_multi_inner=True,
# )