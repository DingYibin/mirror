#!/bin/bash

MEGATRON_PATH=/workspace-dyb/mirror/thirdparty/Megatron-LM
# MEGATRON_PATCH_PATH=/workspace-dyb/mirror/thirdparty/Pai-Megatron-Patch
BASIC_PATH=/workspace-dyb/mirror
export PYTHONPATH=$PYTHONPATH:${BASIC_PATH}:${MEGATRON_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd /workspace-dyb/mirror/thirdparty/Megatron-LM/tools/checkpoint
python convert.py \
    --model-type=GPT \
    --load-dir=/workspace-dyb/experiments/qwq-tp8-dp1/ckpt/0/ \
    --loader=core \
    --saver=hf_llava \
    --save-dir=/workspace-dyb/experiments/qwq-tp8-dp1/converted
