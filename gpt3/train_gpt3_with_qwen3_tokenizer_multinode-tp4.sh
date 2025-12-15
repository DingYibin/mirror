#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# MEGATRON_PATH=/workspace-dyb/mirror/thirdparty/Megatron-LM
# MEGATRON_PATCH_PATH=/workspace-dyb/mirror/thirdparty/Pai-Megatron-Patch
BASIC_PATH=/workspace-dyb/mirror
export PYTHONPATH=$PYTHONPATH:${BASIC_PATH}

GPUS_PER_NODE=$HOST_GPU_NUM
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
NUM_NODES=$HOST_NUM
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
DATA_PATH=$3 #<Specify path and file prefix>_text_document
MODEL_SIZE=$4
TRAIN_MTP_ONLY=$5
export MTP_EH_PROJ_MODE=$6
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)
EXTRA_ARGS=" "
if [ $MODEL_SIZE = 0.6B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    VOCAB_SIZE=151936
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    TP=1
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
    EXTRA_ARGS="--qk-layernorm"
elif [ $MODEL_SIZE = 1.7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=6144
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    VOCAB_SIZE=151936
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    TP=1
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
    EXTRA_ARGS="--qk-layernorm"
elif [ $MODEL_SIZE = 8B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=12288
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    VOCAB_SIZE=151936
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    TP=2
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
    EXTRA_ARGS=" --untie-embeddings-and-output-weights --qk-layernorm"
elif [ $MODEL_SIZE = QWQ32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=27648
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=131072
    EXTRA_VOCAB_SIZE=421
    VOCAB_SIZE=151936
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-5
    TP=4
    TOEKENIZER_MODEL="/public/llm_models/Qwen/QwQ-32B"
    MTP_NUM_LAYERS=7
    if [ $MTP_EH_PROJ_MODE = 10 ]; then
        MTP_NUM_LAYERS=14
    fi
    EXTRA_ARGS=" \
        --untie-embeddings-and-output-weights \
        --add-qkv-bias \
        --rotary-seq-len-interpolation-factor 1 \
        --mtp-num-layers ${MTP_NUM_LAYERS} \
        --main-model-checkpoint /workspace-dyb/qwen-ckpts/QwQ-32B-hf-to-mcore-te-tp4-pp1/release \
        --mtp-loss-scaling-factor 1.0 \
    "
    
else
    echo "MODEL_SIZE=${MODEL_SIZE} is not supported, will be set as 0.6B"
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    VOCAB_SIZE=151936
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    TP=1
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
    EXTRA_ARGS="--qk-layernorm"
fi


NUM_TRAIN_ITERATIONS=16384
SAVE_INTERVAL=8192
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=4
if [ $NUM_NODES = 1 ]; then
    NUM_TRAIN_ITERATIONS=3072
    SAVE_INTERVAL=1024
    GLOBAL_BATCH_SIZE=32
fi
if [ $MTP_EH_PROJ_MODE = 10 ]; then
    MICRO_BATCH_SIZE=2
fi
TRAINING_ARGS=(
    --micro-batch-size 2 
    --global-batch-size 32 
    # --rampup-batch-size 16 16 5859375 
    # --train-samples 262144
    --train-iters $NUM_TRAIN_ITERATIONS
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.008 
    --clip-grad 1.0 
    --bf16
    --lr 1.0e-4 
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .001 
    --lr-decay-iters 3000 
    --no-gradient-accumulation-fusion
    --seq-length 4096
)

if [ $TRAIN_MTP_ONLY = 1 ]; then
    EXTRA_ARGS=${EXTRA_ARGS}" --train-mtp-only --convert-checkpoint"
    if [ $MTP_EH_PROJ_MODE = 0 ]; then
        EXTRA_ARGS=${EXTRA_ARGS}" --lr-decay-style constant"
    else
        NUM_DECAY_ITERATIONS=$((NUM_TRAIN_ITERATIONS/2))
        EXTRA_ARGS=${EXTRA_ARGS}" --lr-decay-style cosine  --lr-decay-iters "${NUM_DECAY_ITERATIONS}
    fi
    TRAINING_ARGS=(
        --micro-batch-size $MICRO_BATCH_SIZE 
        --global-batch-size $GLOBAL_BATCH_SIZE 
        # --rampup-batch-size 16 16 5859375 
        # --train-samples 262144
        --train-iters $NUM_TRAIN_ITERATIONS
        --weight-decay 0.1 
        --adam-beta1 0.9 
        --adam-beta2 0.95 
        --init-method-std 0.008 
        --clip-grad 1.0 
        --bf16
        --lr 1.0e-4 
        --min-lr 1.0e-5
        --lr-warmup-iters 1024
        --no-gradient-accumulation-fusion
        --seq-length 4096
    )
fi

GPT_MODEL_ARGS=(
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --no-masked-softmax-fusion
    --disable-bias-linear
    --position-embedding-type rope
    --no-rope-fusion
    --normalization RMSNorm
    --swiglu
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $INTERMEDIATE_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --group-query-attention
    --num-query-groups $NUM_KEY_VALUE_HEADS
    --kv-channels 128
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --use-rotary-position-embeddings
    --rotary-percent 1.0
    --rotary-base $ROPE_THETA
    --norm-epsilon ${RMS_NORM_EPS}
    --no-bias-swiglu-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
)



MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type HuggingFaceTokenizer
    --make-vocab-size-divisible-by 1188
    --tokenizer-model $TOEKENIZER_MODEL
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval $SAVE_INTERVAL
    --eval-interval 16384 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} gpt3/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    $EXTRA_ARGS
