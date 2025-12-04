#!/bin/bash

LOGS_FILE=single-$(date +%Y-%m%d-%H%M-%S).log
echo $LOGS_FILE
gpt3/train_gpt3_with_qwen3_tokenizer.sh \
    /private/experiments/qwq-tp8-dp1/ckpt \
    /private/experiments/qwq-tp8-dp1/log \
    /private/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data_text_document \
    QWQ32B \
    1 \
    1 \
    &> $LOGS_FILE