import os
from transformers import AutoTokenizer
import json
import jsonlines
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507")

original_dataset = "/public/dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl"

target_dataset = "/workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl"

files = [item for item in os.listdir(original_dataset) if item.endswith("jsonl")]

os.makedirs(target_dataset, exist_ok=True)

odata = []
for file in files:
    sourcefile = os.path.join(original_dataset, file)
    print(sourcefile)
    with open(sourcefile, 'r', encoding='utf-8') as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            try:
                line = line.strip()
                if not line:
                    continue
                idata = json.loads(line)
                idata = idata["conversation"]
                converseion = ""
                for one in idata:
                    converseion += "human:\n"
                    converseion += one['human'] + "\n\n"
                    converseion += "assistant:\n"
                    converseion += one['assistant'] + "\n\n"
                # converseion_text = json.dumps({"text": converseion_text}, ensure_ascii=False)
                # print(f"{converseion=}\n")
                # break
                odata.append({"text": converseion})
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析错误: {e}")
                continue
targetfile = os.path.join(target_dataset, "data.jsonl")
with jsonlines.open(targetfile, mode='w') as t:
    t.write_all(odata)

