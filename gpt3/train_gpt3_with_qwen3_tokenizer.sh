#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

export HOST_GPU_NUM=8
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export HOST_NUM=1
export NODE_RANK=0

gpt3/train_gpt3_with_qwen3_tokenizer_multinode.sh $1 $2 $3 $4 $5