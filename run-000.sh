source /workspace-dyb/mirror/.venv/bin/activate
which python
echo MASTER_ADDR=localhost
echo MASTER_PORT=6000
echo NODE_RANK=0
echo HOST_NUM=1
echo HOST_GPU_NUM=8
LOGS_FILE=/workspace-dyb/mirror/logs/logs-tp4-dp2/s-$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE

EXPERIMENTS_DIR=/workspace-dyb/experiments

# experiment
EXPERIMENT_NAME=qwq-tp4-dp2-mtp-mode-5
MTP_EH_PROJ_MODE=5

gpt3/train_gpt3_with_qwen3_tokenizer_multinode-tp4.sh \
    ${EXPERIMENTS_DIR}/ckpt/${EXPERIMENT_NAME} \
    ${EXPERIMENTS_DIR}/log/${EXPERIMENT_NAME} \
    /workspace-dyb/converted_dataset/a-m-team/merged-r1-dataset \
    QWQ32B \
    1 \
    ${MTP_EH_PROJ_MODE} \
    &> $LOGS_FILE