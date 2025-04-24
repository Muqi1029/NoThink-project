#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6
python main.py \
    --model-path DeepSeek-R1-Distill-Qwen-7B \
    --backend sglang \
    --debug \
    --dataset gpqa-diamond
