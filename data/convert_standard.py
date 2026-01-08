import os


source_dir = "/public/dataset/generate_from_dpsk_r1"
target_dir = "/workspace-dyb/converted_dataset/generate_from_dpsk_r1/split"
files = [file for file in os.listdir(source_dir) if file.endswith('.jsonl')]
# print(files)

for file in files:
    source_file = os.path.join(source_dir, file)
    target_file = os.path.join(target_dir, file)
    # model = file.split('_')[1]
    # if model == 'r1':
    print(file)
    cmd = "python /workspace-dyb/mirror/thirdparty/Megatron-LM/tools/preprocess_data.py"
    cmd += f" --input {source_file}"
    cmd += f" --output-prefix {target_file[:-len('.jsonl')]}"
    cmd += " --tokenizer-type HuggingFaceTokenizer"
    cmd += " --tokenizer-model /public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507"
    cmd += " --append-eod --workers 48"
    os.system(cmd)
    # print(cmd)



