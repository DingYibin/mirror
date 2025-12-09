
uv run python /workspace-dyb/mirror/thirdparty/Megatron-LM/tools/preprocess_data.py \
    --input /workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/data.jsonl \
    --output-prefix /workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --append-eod \
    --workers 8 



