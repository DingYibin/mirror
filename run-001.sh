source /workspace-dyb/mirror/.venv/bin/activate
which python
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo NODE_RANK=$NODE_RANK
echo HOST_NUM=$HOST_NUM
echo HOST_GPU_NUM=$HOST_GPU_NUM
LOGS_FILE=/workspace-dyb/mirror/logs/logs-text/log_$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE

gpt3/train_gpt3_with_qwen3_tokenizer_multinode-tp4.sh \
    /workspace-dyb/experiments/ckpt/qwq-tp4-dp4-mtp-mode-5 \
    /workspace-dyb/experiments/log/qwq-tp4-dp4-mtp-mode-5 \
    /workspace-dyb/converted_dataset/a-m-team/merged-r1-dataset \
    QWQ32B \
    1 \
    5 \
    &> $LOGS_FILE & tail -f $LOGS_FILE