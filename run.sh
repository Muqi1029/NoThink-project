#!/usr/bin/env bash

dataset=gpqa-diamond
categories="Think" # "Think NoThink ThinkOver"


export CUDA_VISIBLE_DEVICES=1
python main.py \
    --backend sglang \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --dataset ${dataset} \
    --categories ${categories} \
    --num-samples 1
