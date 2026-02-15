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
PROVIDER=""  # Set to "deepinfra/fp8" or leave empty for OpenRouter default routing

# Model configurations
declare -A MODELS
MODELS["qwen_qwen3_vl_8b_thinking"]="qwen/qwen3-vl-8b-thinking"

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

    # Build provider argument
    PROVIDER_ARG=""
    if [ -n "$PROVIDER" ]; then
        PROVIDER_ARG="--provider $PROVIDER"
    fi

    # System prompt attack
    echo "Running system prompt attack..."
    python src/inference/system_prompt.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        $PROVIDER_ARG \
        --output-dir "results/${MODEL_NAME}/system_prompts"

    # Assistant prefill attack
    # Note: This will generate multiple files: ${MODEL_NAME}_assistant_prefill_prefill_0.json, etc.
    echo "Running assistant prefill attack..."
    python src/inference/assistant_prefill_attack.py \
        --questions "$PREFILL_QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --prefill-type answer \
        $PROVIDER_ARG \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_assistant_prefill.json"

    # User prefill attack
    # Note: This will generate multiple files: ${MODEL_NAME}_user_prefill_prefill_0.json, etc.
    echo "Running user prefill attack..."
    python src/inference/user_prefill_attack.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples 5 \
        --num-initial-samples 5 \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        $PROVIDER_ARG \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_user_prefill.json"

    # User prefill simple attack
    echo "Running user prefill simple attack..."
    python src/inference/user_prefill_simple_attack.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        $PROVIDER_ARG \
        --output "results/${MODEL_NAME}/${MODEL_NAME}_user_prefill_simple.json"

    # Pretrain prompt attack
    echo "Running pretrain prompt attack..."
    python src/inference/pretrain_prompt_attack.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        $PROVIDER_ARG \
        --output-dir "results/${MODEL_NAME}/pretrain_prompts"

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

echo ""
echo "=========================================="
echo "GENERATING EVALUATION CONFIGS"
echo "=========================================="
echo ""

cd /root/chinese-model-honesty
python scripts/generate_attack_eval_configs.py

echo ""
echo "=========================================="
echo "RUNNING EVALUATIONS"
echo "=========================================="
echo ""

cd /root/chinese_auditing

# Count total configs
TOTAL_CONFIGS=$(ls /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen3_vl_8b_thinking_*.yaml 2>/dev/null | wc -l)
CURRENT=0

# Run evaluations for all generated configs
for config in /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen3_vl_8b_thinking_*.yaml; do
    if [ -f "$config" ]; then
        CURRENT=$((CURRENT + 1))
        config_name=$(basename "$config" .yaml)

        echo ""
        echo "[$CURRENT/$TOTAL_CONFIGS] Evaluating: $config_name"
        echo "----------------------------------------"

        python src/evaluation/evaluation_pipeline.py "$config"
        eval_exit=$?

        if [ $eval_exit -eq 0 ]; then
            echo "✓ Evaluation complete: $config_name"
        else
            echo "✗ Evaluation failed: $config_name (exit code: $eval_exit)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Evaluation results saved in: /root/chinese_auditing/output/evaluation/qwen3_vl_8b_thinking/"
