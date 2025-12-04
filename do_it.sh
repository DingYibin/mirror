source /private/mirror/.venv/bin/activate
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo NODE_RANK=$NODE_RANK
echo HOST_NUM=$HOST_NUM
echo HOST_GPU_NUM=$HOST_GPU_NUM
LOGS_FILE=log_$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE
# gpt3/train_gpt3_with_qwen3_tokenizer_multinode.sh \
#     /private/experiments/gpt-tp8-dp2/ckpt \
#     /private/experiments/gpt-tp8-dp2/logs \
#     /private/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data_text_document \
#     &> $LOGS_FILE

gpt3/train_gpt3_with_qwen3_tokenizer_multinode.sh \
    /private/experiments/qwq-tp8-dp2/ckpt \
    /private/experiments/qwq-tp8-dp2/log \
    /private/converted_dataset/a-m-team/merged-r1-dataset \
    QWQ32B \
    1 \
    0 \
    &> $LOGS_FILE & tail -f $LOGS_FILE