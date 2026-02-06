#!/bin/bash

source /root/.venv/bin/activate

cd /root/chinese-model-honesty

CONFIG="honesty_training/input/control_chinese_topics.yaml"

# Base model
BASE_MODEL="Qwen/Qwen3-32B"

# Output directory (must match the yaml config)
OUTPUT_DIR="/workspace/qwen3-32b-lora-finetuned-chinese-censored-gpt"

# Lora adapter name for output files
LORA_NAME="qwen3-32b-chinese-censored-gpt"

# Questions file
QUESTIONS="data/dev_questions.json"

echo "=========================================="
echo "Training model with config: $CONFIG"
echo "=========================================="
python honesty_training/finetune_qwen3_32b.py "$CONFIG"

echo "=========================================="
echo "Evaluating finetuned model"
echo "=========================================="
python honesty_eval/sample_assistant_responses_local.py \
    --model "$BASE_MODEL" \
    --questions "$QUESTIONS" \
    --output "results/honesty/${LORA_NAME}.json" \
    --temperature 0.7 \
    --num-samples 10 \
    --batch-size 200 \
    --max-tokens 2048 \
    --tensor-parallel-size 1 \
    --lora "$OUTPUT_DIR"

echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to results/honesty/${LORA_NAME}.json"
echo "=========================================="
