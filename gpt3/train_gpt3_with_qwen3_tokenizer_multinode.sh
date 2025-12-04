#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# MEGATRON_PATH=/private/mirror/thirdparty/Megatron-LM
# MEGATRON_PATCH_PATH=/private/mirror/thirdparty/Pai-Megatron-Patch
BASIC_PATH=/private/mirror
export PYTHONPATH=$PYTHONPATH:${BASIC_PATH}

GPUS_PER_NODE=$HOST_GPU_NUM
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
NUM_NODES=$HOST_NUM
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1/$NODE_RANK #<Specify path>
TENSORBOARD_LOGS_PATH=$2/$NODE_RANK #<Specify path>
DATA_PATH=$3 #<Specify path and file prefix>_text_document
MODEL_SIZE=$4
TRAIN_MTP_ONLY=$5
export MTP_MOVE_EH_PROJ=$6
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
    TP=8
    TOEKENIZER_MODEL="/public/llm_models/Qwen/QwQ-32B"
    EXTRA_ARGS=" \
        --untie-embeddings-and-output-weights \
        --add-qkv-bias \
        --rotary-seq-len-interpolation-factor 1 \
        --mtp-num-layers 7 \
        --main-model-checkpoint /private/qwen-ckpts/QwQ-32B-hf-to-mcore-te-tp8-pp1/release \
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


NUM_TRAIN_ITERATIONS=32768
if [ $NUM_NODES = 1 ]; then
    NUM_TRAIN_ITERATIONS=4096
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
    EXTRA_ARGS=${EXTRA_ARGS}" --train-mtp-only "
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
        --lr-decay-style constant 
        --min-lr 1.0e-5
        --lr-warmup-iters 500
        # --lr-decay-iters 3000 
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
    --save-interval 1000
    --eval-interval 10000 
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
