import os
import torch

source_dir = "/workspace-dyb/experiments/qwq-tp8-dp1/ckpt/0/converted"
target_dir = "/workspace-dyb/experiments/qwq-tp8-dp1/ckpt/0/converted-hf"

source_dir = "/workspace-dyb/experiments/qwq-tp8-dp2/ckpt/converted-0"
target_dir = "/workspace-dyb/experiments/qwq-tp8-dp2/ckpt/converted-0-hf"

source_dir = "/workspace-dyb/experiments/qwq-tp8-dp2-move/ckpt/converted-0"
target_dir = "/workspace-dyb/experiments/qwq-tp8-dp2-move/ckpt/converted-0-hf"

tp_size = 8
pp_size = 1

num_attention_heads = 40
num_key_value_heads = 8


weights = [None] * tp_size

keys = set()

for i in range(tp_size):
    for j in range(pp_size):
        filepath = os.path.join(source_dir, f"tp-{tp_size}-{i}-pp-{pp_size}-{j}.pt")
        one_weight = torch.load(filepath)
        if weights[i] is None:
            weights[i] = {}
        for key, value in one_weight.items():
            if key.startswith('mtp'):
                weights[i][key] = value.cpu()
                # print(key, value.shape if hasattr(value, 'shape') else value)
        del one_weight
        # print(weights[i].keys())
        
    keys.update(set(weights[i].keys()))

# exit()
keys = list(keys)
keys.sort()
# print(keys)
merged_weight = dict()
for key in keys:
    if "_extra_state" in key:
        continue
    if "norm" in key:
        for i in range(tp_size):
            val = weights[i].get(key)
            if val is not None:
                merged_weight[key] = val
                break
    elif key.endswith(("eh_proj.weight", "linear_fc1.weight", "self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias")):
        # column parallel
        vals = [weights[i][key] for i in range(tp_size)]
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
            out_str += f"\n{val}"
        print(out_str)

converted_weight = dict()
for key, val in merged_weight.items():
    name = key.split(".")
    if name[3] == 'transformer_layer':
        name = name[:3] + name[4:]
    if name[3] == "self_attention":
        name[3] = "self_attn"
    if len(name) >= 6 and name[5] == 'layer_norm_weight':
        if name[3] == 'self_attn' and name[4] == 'linear_qkv':
            name = name[:3] + ['input_layernorm', 'weight']
        elif name[3] == 'mlp' and name[4] == 'linear_fc1':
            name = name[:3] + ['post_attention_layernorm', 'weight']
    elif name[3] == 'mlp' and name[4] == 'linear_fc2':
        name[4] = 'down_proj'   
    elif name[3] == 'self_attn' and name[4] == 'linear_proj':
        name[4] = 'o_proj'
    elif name[3] == 'final_layernorm':
        name = name[:3] + ['shared_head', 'norm', 'weight']

    if name[3] == 'mlp' and name[4] == 'linear_fc1':
        gate_proj, up_proj = torch.chunk(val, 2, 0)

        name[4] = 'gate_proj'
        gate_key = ".".join(name)
        converted_weight[gate_key] = gate_proj

        name[4] = 'up_proj'
        up_key = ".".join(name)
        converted_weight[up_key] = up_proj

    elif name[3] == 'self_attn' and name[4] == 'linear_qkv':
        # torch.split(val, )
        dim = val.shape[0]
        head_dim = dim // (num_attention_heads + num_key_value_heads * 2)
        q_dim = head_dim * num_attention_heads
        k_dim = head_dim * num_key_value_heads
        v_dim = head_dim * num_key_value_heads
        q_proj, k_proj, v_proj = torch.split(val, [q_dim, k_dim, v_dim])
        
        name[4] = "q_proj"
        q_key = ".".join(name)
        converted_weight[q_key] = q_proj
        
        name[4] = "k_proj"
        k_key = ".".join(name)
        converted_weight[k_key] = k_proj
    
        name[4] = "v_proj"
        v_key = ".".join(name)
        converted_weight[v_key] = v_proj
        
    else:
        key = ".".join(name)
        
        converted_weight[key] = val
del merged_weight

split_weights = dict()
save_info = dict()
for key, val in converted_weight.items():
    layer_id = key.split(".")[2]
    if layer_id not in split_weights:
        split_weights[layer_id] = dict()
    split_weights[layer_id][key] = val

from safetensors.torch import save_file
import json
os.makedirs(target_dir, exist_ok=True)
total_size = 0
weight_map = dict()
for i, (layer_id, layer_weights) in enumerate(split_weights.items()):
    print(layer_id)
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


