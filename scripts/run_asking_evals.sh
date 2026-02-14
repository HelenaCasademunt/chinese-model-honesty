#!/bin/bash

source /root/.venv/bin/activate

# Qwen model
QWEN_INPUT="results/evaluated_qwen_qwen3_32b_baseline_no_thinking.json"
QWEN_MODEL="qwen/qwen3-32b"
QWEN_OUTPUT_DIR="results/asking_model/qwen3_32b"

mkdir -p "$QWEN_OUTPUT_DIR"

echo "=== Running evaluations for Qwen3-32B ==="

# echo "Running ask_if_true.py..."
# python asking_the_model/ask_if_true.py \
#     --input "$QWEN_INPUT" \
#     --output "$QWEN_OUTPUT_DIR/ask_if_true.json" \
#     --model "$QWEN_MODEL"

# echo "Running ask_if_true_confession.py..."
# python asking_the_model/ask_if_true_confession.py \
#     --input "$QWEN_INPUT" \
#     --output "$QWEN_OUTPUT_DIR/confession.json" \
#     --model "$QWEN_MODEL"

echo "Running evaluate_confession.py..."
python asking_the_model/evaluate_confession.py \
    --input "$QWEN_OUTPUT_DIR/confession.json" \
    --output "$QWEN_OUTPUT_DIR/confession_evaluated.json"

# DeepSeek model
DEEPSEEK_INPUT="results/evaluated_deepseek_deepseek_r1_distill_llama_70b_baseline_no_thinking.json"
DEEPSEEK_MODEL="deepseek/deepseek-r1-distill-llama-70b"
DEEPSEEK_OUTPUT_DIR="results/asking_model/deepseek_r1_70b"

mkdir -p "$DEEPSEEK_OUTPUT_DIR"

echo ""
echo "=== Running evaluations for DeepSeek-R1-70B ==="

echo "Running ask_if_true.py..."
python asking_the_model/ask_if_true.py \
    --input "$DEEPSEEK_INPUT" \
    --output "$DEEPSEEK_OUTPUT_DIR/ask_if_true.json" \
    --model "$DEEPSEEK_MODEL"

echo "Running ask_if_true_confession.py..."
python asking_the_model/ask_if_true_confession.py \
    --input "$DEEPSEEK_INPUT" \
    --output "$DEEPSEEK_OUTPUT_DIR/confession.json" \
    --model "$DEEPSEEK_MODEL"

echo "Running evaluate_confession.py..."
python asking_the_model/evaluate_confession.py \
    --input "$DEEPSEEK_OUTPUT_DIR/confession.json" \
    --output "$DEEPSEEK_OUTPUT_DIR/confession_evaluated.json"

echo ""
echo "=== All evaluations complete ==="
