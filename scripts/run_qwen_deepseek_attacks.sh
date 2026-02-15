#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

# Common parameters
QUESTIONS="data/dev_questions.json"
PREFILL_QUESTIONS="src/inference/prompts/assistant_prefill_dev_questions.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000

# Model configurations
declare -A MODELS
MODELS["qwen_qwen3_vl_8b_instruct"]="qwen/qwen3-vl-8b-instruct"
MODELS["deepseek_deepseek_r1_distill_qwen_32b"]="deepseek/deepseek-r1-distill-qwen-32b"

# Create base results directories
mkdir -p results/system_prompts results/pretrain_prompts

# Run attacks for each model
for MODEL_NAME in "${!MODELS[@]}"; do
    MODEL="${MODELS[$MODEL_NAME]}"

    echo ""
    echo "=========================================="
    echo "Running attacks for: $MODEL"
    echo "Model name: $MODEL_NAME"
    echo "=========================================="
    echo ""

    # Create model-specific result directories
    mkdir -p "results/${MODEL_NAME}"

    # System prompt attack
    echo "Running system prompt attack..."
    python src/inference/system_prompt.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_system_prompt.json"

    # Assistant prefill attack
    echo "Running assistant prefill attack..."
    python src/inference/assistant_prefill_attack.py \
        --questions "$PREFILL_QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --prefill-type answer \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_assistant_prefill.json"

    # User prefill attack
    echo "Running user prefill attack..."
    python src/inference/user_prefill_attack.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples 5 \
        --num-initial-samples 5 \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_user_prefill.json"

    # User prefill simple attack
    echo "Running user prefill simple attack..."
    python src/inference/user_prefill_simple_attack.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_user_prefill_simple.json"

    # Pretrain prompt attack
    echo "Running pretrain prompt attack..."
    python src/inference/pretrain_prompt_attack.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_pretrain_prompt.json"

    echo ""
    echo "✓ Completed attacks for $MODEL"
    echo ""
done

echo ""
echo "=========================================="
echo "All attacks complete!"
echo "=========================================="
echo ""
echo "Results saved in:"
for MODEL_NAME in "${!MODELS[@]}"; do
    echo "  - results/${MODEL_NAME}/"
done
