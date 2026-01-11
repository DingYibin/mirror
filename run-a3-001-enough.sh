source /public/workspace/dyb/mirror/.venv/bin/activate
which python
echo MASTER_ADDR=$MASTER_ADDR
echo MASTER_PORT=$MASTER_PORT
echo NODE_RANK=$NODE_RANK
echo HOST_NUM=$HOST_NUM
echo HOST_GPU_NUM=$HOST_GPU_NUM
LOGS_FILE=/public/workspace/dyb/mirror/logs/logs-text/log_$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE

EXPERIMENTS_DIR=/workspace-dyb/experiments

# experiment
EXPERIMENT_NAME=qwen-30b-a3b-thinking-tp1-ep8-enough
MTP_EH_PROJ_MODE=0
JUST_CONVERT_CKPT=0

gpt3/train_gpt3_with_qwen3_tokenizer_multinode-dp8-ep8.sh \
    ${EXPERIMENTS_DIR}/ckpt/${EXPERIMENT_NAME} \
    ${EXPERIMENTS_DIR}/log/${EXPERIMENT_NAME} \
    /workspace-dyb/converted_dataset/all \
    A3B \
    2 \
    ${MTP_EH_PROJ_MODE} \
    131072 \
    8192 \
    ${JUST_CONVERT_CKPT} \
    &> $LOGS_FILE