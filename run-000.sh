source /public/workspace/dyb/mirror/.venv/bin/activate
which python
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NODE_RANK=0
export HOST_NUM=1
export HOST_GPU_NUM=8
# export CUDA_VISIBLE_DEVICES=4,5,6,7
LOGS_FILE=/public/workspace/dyb/mirror/logs/logs/dev/s-$(date +%Y-%m%d-%H%M-%S)_${NODE_RANK}_${HOST_NUM}.log
echo $LOGS_FILE

EXPERIMENTS_DIR=/workspace-dyb/experiments

# experiment
EXPERIMENT_NAME=qwen3-a3b-thinking-tp1-ep8-dev
MTP_EH_PROJ_MODE=0

# export NUM_TRANSFORMER_BLOCK_ONE_MTP_LAYER=2
gpt3/train_gpt3_with_qwen3_tokenizer_multinode-dp8-ep8.sh \
    ${EXPERIMENTS_DIR}/ckpt/${EXPERIMENT_NAME} \
    ${EXPERIMENTS_DIR}/log/${EXPERIMENT_NAME} \
    /workspace-dyb/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data_text_document \
    A3B \
    1 \
    ${MTP_EH_PROJ_MODE} \
    3072 \
    1024 \
    0 \
    &> $LOGS_FILE