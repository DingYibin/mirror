source /workspace-dyb/mirror/.venv/bin/activate
which python
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo NODE_RANK=$NODE_RANK
echo HOST_NUM=$HOST_NUM
echo HOST_GPU_NUM=$HOST_GPU_NUM
LOGS_FILE=/workspace-dyb/mirror/logs/logs-text/log_$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE

EXPERIMENTS_DIR=/workspace-dyb/experiments

# experiment
EXPERIMENT_NAME=qwq-tp4-dp4-mtp-mode-9
MTP_EH_PROJ_MODE=9

gpt3/train_gpt3_with_qwen3_tokenizer_multinode-tp4.sh \
    ${EXPERIMENTS_DIR}/ckpt/${EXPERIMENT_NAME} \
    ${EXPERIMENTS_DIR}/log/${EXPERIMENT_NAME} \
    /workspace-dyb/converted_dataset/a-m-team/merged-r1-dataset \
    QWQ32B \
    1 \
    ${MTP_EH_PROJ_MODE} \
    &> $LOGS_FILE