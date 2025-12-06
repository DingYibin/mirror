#!/bin/bash

MEGATRON_PATH=/private/mirror/thirdparty/Megatron-LM
# MEGATRON_PATCH_PATH=/private/mirror/thirdparty/Pai-Megatron-Patch
BASIC_PATH=/private/mirror
export PYTHONPATH=$PYTHONPATH:${BASIC_PATH}:${MEGATRON_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd /private/mirror/thirdparty/Megatron-LM/tools/checkpoint
python convert.py \
    --model-type=GPT \
    --load-dir=/private/experiments/qwq-tp8-dp1/ckpt/0/ \
    --loader=core \
    --saver=hf_llava \
    --save-dir=/private/experiments/qwq-tp8-dp1/converted
