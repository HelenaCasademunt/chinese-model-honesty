#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

MODEL="deepseek/deepseek-r1-distill-llama-70b"
MODEL_NAME="deepseek_deepseek_r1_distill_llama_70b"
QUESTIONS="data/dev_questions.json"
PREFILL_QUESTIONS="src/inference/prompts/assistant_prefill_dev_questions.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000

mkdir -p results results/system_prompts results/pretrain_prompts

# System prompt attack
python src/inference/system_prompt.py \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"

# Assistant prefill attack
python src/inference/assistant_prefill_attack.py \
    --questions "$PREFILL_QUESTIONS" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --prefill-type answer \
    --output "results/${MODEL_NAME}_assistant_prefill.json"

# User prefill attack
python src/inference/user_prefill_attack.py \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --num-samples 5 \
    --num-initial-samples 5 \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --output "results/${MODEL_NAME}_user_prefill.json"

# User prefill simple attack
python src/inference/user_prefill_simple_attack.py \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --output "results/${MODEL_NAME}_user_prefill_simple.json"

# Pretrain prompt attack
python src/inference/pretrain_prompt_attack.py \
    --questions "$QUESTIONS" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"

echo "All attacks complete."

# Evaluations
echo ""
echo "=========================================="
echo "RUNNING EVALUATIONS"
echo "=========================================="

EVAL_CONFIGS_DIR="configs/deepseek_attacks"

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
echo "All evaluations complete."
