#!/bin/bash

source /root/.venv/bin/activate

python honesty_training/finetune_qwen3_32b.py \
    --data honesty_training/goals-data-qwen3.jsonl \
    --output-dir honesty_training/qwen3-32b-lora-finetuned \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-4 \
    --max-seq-length 4096 \
    --lora-r 32 \
    --lora-alpha 64 \
    --save-steps 100
