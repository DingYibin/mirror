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
EXPERIMENT_NAME=qwq-mode-10-2-1220
MTP_EH_PROJ_MODE=10
export NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER=2
JUST_CONVERT_CKPT=1

gpt3/train_gpt3_with_qwen3_tokenizer_multinode-dp8-ep8.sh \
    ${EXPERIMENTS_DIR}/ckpt/${EXPERIMENT_NAME} \
    ${EXPERIMENTS_DIR}/log/${EXPERIMENT_NAME} \
    /workspace-dyb/converted_dataset/a-m-team/merged-r1-dataset \
    QWQ32B \
    1 \
    ${MTP_EH_PROJ_MODE} \
    ${JUST_CONVERT_CKPT} \
    &> $LOGS_FILE