#!/bin/bash

LOGS_FILE=single-$(date +%Y-%m%d-%H%M-%S).log
echo $LOGS_FILE
gpt3/train_gpt3_with_qwen3_tokenizer.sh \
    /workspace-dyb/experiments/qwq-tp8-dp1/ckpt \
    /workspace-dyb/experiments/qwq-tp8-dp1/log \
    /workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data_text_document \
    QWQ32B \
    1 \
    1 \
    &> $LOGS_FILE