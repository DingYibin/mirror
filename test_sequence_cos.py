from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

model_path = "/public/llm_models/Qwen/QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
data_path = "/workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/data.jsonl"
with open(data_path, 'r', encoding='utf-8') as f:
    for item in f.readlines():
        data = json.loads(item)
        break

text = data['text']

inputs = torch.tensor(tokenizer.encode(text)).cuda()

dis = torch.matmul(output, output.T)
dis_diag_rsqrt = dis.diag().rsqrt()
dis_norm = dis_diag_rsqrt[:, None] * dis * dis_diag_rsqrt[None, :]

import numpy as np


data = dis_norm.cpu().detach().numpy()
plt.figure(figsize=(6, 5))
# 绘制热力图，'coolwarm'颜色映射适合表示从负到正的数据[citation:7]
im = plt.imshow(data, cmap='coolwarm')
plt.colorbar(im)  # 添加颜色条
plt.title("Cosine Distance Heatmap")
plt.show()