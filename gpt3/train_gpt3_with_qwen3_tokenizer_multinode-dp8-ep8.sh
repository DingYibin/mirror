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
NUM_TRAIN_ITERATIONS=$7
SAVE_INTERVAL=$8
JUST_CONVERT_CKPT=$9

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

MODEL_PARALLEL_ARGS=(
	--pipeline-model-parallel-size 1 
)

GPT_MODEL_ARGS=(
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --no-masked-softmax-fusion
    --disable-bias-linear
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --group-query-attention
    --kv-channels 128
    --use-rotary-position-embeddings
    --rotary-percent 1.0
    --no-bias-swiglu-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --no-rope-fusion
    --no-gradient-accumulation-fusion
)
if [ $JUST_CONVERT_CKPT = 1 ]; then
    GPT_MODEL_ARGS+=(
        --ckpt-convert-format torch
        --ckpt-convert-save ${CHECKPOINT_PATH}
    )
fi

GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=1

if [ $MODEL_SIZE = 0.6B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 28
        --hidden-size 1024
        --ffn-hidden-size 3072
        --num-attention-heads 16
        --num-query-groups 8
        --max-position-embeddings 40960
        --rotary-base 1000000
        --norm-epsilon 1e-6
        --qk-layernorm
    )
    MODEL_PARALLEL_ARGS+=(
        --tensor-model-parallel-size 1
    )
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
elif [ $MODEL_SIZE = 1.7B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 28
        --hidden-size 2048
        --ffn-hidden-size 6144
        --num-attention-heads 16
        --num-query-groups 8
        --max-position-embeddings 40960
        --rotary-base 1000000
        --norm-epsilon 1e-6
        --qk-layernorm
    )
    MODEL_PARALLEL_ARGS+=(
        --tensor-model-parallel-size 1
    )
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
elif [ $MODEL_SIZE = 8B ]; then
    GPT_MODEL_ARGS+=(
        --num-layers 36
        --hidden-size 4096
        --ffn-hidden-size 12288
        --num-attention-heads 32
        --num-query-groups 8
        --max-position-embeddings 40960
        --rotary-base 1000000
        --norm-epsilon 1e-6
        --untie-embeddings-and-output-weights
        --qk-layernorm
    )
    MODEL_PARALLEL_ARGS+=(
        --tensor-model-parallel-size 2
    )
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Instruct-2507
elif [ $MODEL_SIZE = QWQ32B ]; then
    TOEKENIZER_MODEL="/public/llm_models/Qwen/QwQ-32B"
    MTP_NUM_LAYERS=7
    if [ $MTP_EH_PROJ_MODE = 10 ]; then
        MTP_NUM_LAYERS=14
        MICRO_BATCH_SIZE=2
    fi
    GPT_MODEL_ARGS+=(
        --num-layers 64
        --hidden-size 5120
        --ffn-hidden-size 27648
        --num-attention-heads 40
        --num-query-groups 8
        --max-position-embeddings 131072
        --rotary-base 1000000
        --norm-epsilon 1e-5
        --untie-embeddings-and-output-weights
        --add-qkv-bias
        --rotary-seq-len-interpolation-factor 1
        --mtp-num-layers ${MTP_NUM_LAYERS}
        --main-model-checkpoint /workspace-dyb/qwen-ckpts/QwQ-32B-hf-to-mcore-te-tp4-pp1/release
        --mtp-loss-scaling-factor 1.0
        --make-vocab-size-divisible-by 1188
    )
    MODEL_PARALLEL_ARGS+=(
        --tensor-model-parallel-size 4
    )
    # GLOBAL_BATCH_SIZE=$((32*$NUM_NODES))
    
elif [ $MODEL_SIZE = A3B ]; then
    MTP_NUM_LAYERS=7
    MTP_LOSS_SCALING_FACTOR=$(python -c "print(1.0 * ${MTP_NUM_LAYERS} / 3.0)")
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-30B-A3B-Thinking-2507
    GPT_MODEL_ARGS+=(
        --num-layers 48
        --hidden-size 2048
        --ffn-hidden-size 6144
        --num-attention-heads 32
        --untie-embeddings-and-output-weights
        --moe-ffn-hidden-size 768
        --moe-grouped-gemm
        --moe-router-score-function softmax
        --moe-token-dispatcher-type alltoall
        --moe-router-topk 8
        --moe-layer-freq "([1]*48)"
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 0.001
        # --moe-layer-recompute
        --num-experts 128
        --num-query-groups 4
        --qk-layernorm
        --max-position-embeddings 40960
        --mtp-num-layers ${MTP_NUM_LAYERS}
        --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR}
        --main-model-checkpoint /workspace-dyb/qwen-ckpts/Qwen3-30B-A3B-Thinking-2507-tp1-ep8-torch/release
        --main-model-checkpoint-dtype torch
        --make-vocab-size-divisible-by 1187
    )
    MODEL_PARALLEL_ARGS+=(
        --tensor-model-parallel-size 1
        --expert-model-parallel-size 8
    )
    # LOAD_PATH=/workspace-dyb/qwen-ckpts/Qwen3-30B-A3B-Thinking-2507-tp1-ep8
    # GLOBAL_BATCH_SIZE=64
    # GLOBAL_BATCH_SIZE=$((1024*$NUM_NODES))
    MICRO_BATCH_SIZE=2
elif [ $MODEL_SIZE = A22B ]; then
    MTP_NUM_LAYERS=3
    MTP_LOSS_SCALING_FACTOR=$(python -c "print(1.0 * ${MTP_NUM_LAYERS} / 3.0)")
    TOEKENIZER_MODEL=/public/llm_models/Qwen/Qwen3-235B-A22B-Instruct-2507
    GPT_MODEL_ARGS+=(
        --num-layers 94
        --hidden-size 4096
        --ffn-hidden-size 12288
        --num-attention-heads 64
        --untie-embeddings-and-output-weights
        --moe-ffn-hidden-size 1536
        --moe-grouped-gemm
        --moe-router-score-function softmax
        --moe-token-dispatcher-type alltoall
        --moe-router-topk 8
        --moe-layer-freq "([1]*94)"
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 0.001
        # --moe-layer-recompute
        --num-experts 128
        --num-query-groups 4
        --qk-layernorm
        --max-position-embeddings 40960
        --mtp-num-layers ${MTP_NUM_LAYERS}
        --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR}
        --main-model-checkpoint /workspace-dyb/qwen-ckpts/Qwen3-235B-A22B-Instruct-2507-tp1-ep8/release
        --main-model-checkpoint-dtype torch
        --make-vocab-size-divisible-by 1187
    )
    MODEL_PARALLEL_ARGS+=(
        --tensor-model-parallel-size 1
        --expert-model-parallel-size 8
    )
    # GLOBAL_BATCH_SIZE=$((64*$NUM_NODES))
    # MICRO_BATCH_SIZE=1
else
    echo "MODEL_SIZE=${MODEL_SIZE} is not supported"
    exit
fi

NUM_WARMUP_ITERATIONS=$((NUM_TRAIN_ITERATIONS/16))
NUM_DECAY_ITERATIONS=$((NUM_TRAIN_ITERATIONS/2))

TRAINING_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.008 
    --clip-grad 1.0 
    --bf16
    --seq-length 4096
    --lr 1.0e-4 
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-decay-iters ${NUM_DECAY_ITERATIONS}
    --lr-warmup-iters ${NUM_WARMUP_ITERATIONS}
    --train-iters $NUM_TRAIN_ITERATIONS
)


if [ $TRAIN_MTP_ONLY = 1 ]; then
    TRAINING_ARGS+=(
        --train-mtp-only
        --convert-checkpoint
    )
fi


DATA_ARGS=(
    --data-path $DATA_PATH 
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOEKENIZER_MODEL
    --split 949,50,1
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
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
    ${EVAL_AND_LOGGING_ARGS[@]}
