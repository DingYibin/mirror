#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

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
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
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
    EXTRA_ARGS=" --untie-embeddings-and-output-weights "
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
    --qk-layernorm
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --rotary-percent 1.0
    --rotary-base $ROPE_THETA
    --norm-epsilon ${RMS_NORM_EPS}
    --no-bias-swiglu-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 256 
    # --rampup-batch-size 16 16 5859375 
    # --train-samples 262144
    --train-iters 5000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 500 
    --no-gradient-accumulation-fusion
    --seq-length 4096
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1024 \
    --tokenizer-model /public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 1000 
    --eval-interval 1000 
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
