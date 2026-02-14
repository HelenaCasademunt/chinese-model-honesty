#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

MODEL="deepseek/deepseek-r1-distill-qwen-14b"
MODEL_NAME="deepseek_deepseek_r1_distill_qwen_14b"
QUESTIONS="data/dev_questions.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000

mkdir -p results

# Baseline (with thinking)
echo "=========================================="
echo "Running baseline (with thinking)"
echo "=========================================="
python src/inference/baseline.py \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --output "results/${MODEL_NAME}_baseline.json"

# Baseline no thinking
echo ""
echo "=========================================="
echo "Running baseline (no thinking)"
echo "=========================================="
python src/inference/baseline_no_thinking.py \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --output "results/${MODEL_NAME}_no_thinking.json"

echo ""
echo "=========================================="
echo "RUNNING EVALUATIONS"
echo "=========================================="

EVAL_CONFIGS_DIR="configs/deepseek_distill_qwen_14b"

for config in "$EVAL_CONFIGS_DIR"/*.yaml; do
    [ -f "$config" ] || continue
    config_name=$(basename "$config" .yaml)

    echo ""
    echo "=========================================="
    echo "Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/evaluation_pipeline.py "$config"
done

echo ""
echo "All done."
