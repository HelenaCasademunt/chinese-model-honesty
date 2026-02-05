#!/bin/bash

# Run all evaluations for Qwen and DeepSeek models
# Usage: ./run_all_evals.sh [--model qwen|deepseek|both] [--attack <name>]

source /root/.venv/bin/activate

# Default settings
MODELS="both"
ATTACK=""

# Model identifiers
QWEN_MODEL="qwen/qwen3-32b"
DEEPSEEK_MODEL="deepseek/deepseek-r1-distill-llama-70b"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS="$2"
            shift 2
            ;;
        --attack)
            ATTACK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build list of models to run
declare -a MODEL_LIST
case $MODELS in
    qwen)
        MODEL_LIST=("$QWEN_MODEL")
        ;;
    deepseek)
        MODEL_LIST=("$DEEPSEEK_MODEL")
        ;;
    both)
        MODEL_LIST=("$QWEN_MODEL" "$DEEPSEEK_MODEL")
        ;;
    *)
        echo "Unknown model: $MODELS (use qwen, deepseek, or both)"
        exit 1
        ;;
esac

# Evaluation settings
FACTS_PATH="data/dev_facts_with_ids.json"
EVALUATOR_MODEL="google/gemini-3-flash-preview"
HONESTY_MODEL="openai/gpt-5.2"
EVAL_CONCURRENT=10

eval_baseline() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_file="results/${model_name}_baseline.json"
    local output_file="results/evaluated_${model_name}_baseline.json"

    if [[ -f "$output_file" ]]; then
        echo "Skipping eval baseline for $model (output exists: $output_file)"
        return 0
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file - skipping evaluation"
        return 1
    fi

    echo "============================================================"
    echo "Evaluating baseline for $model"
    echo "============================================================"
    python honesty_eval/evaluate.py \
        --attack-type baseline \
        --input "$input_file" \
        --output "$output_file" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --honesty-model "$HONESTY_MODEL" \
        --max-concurrent "$EVAL_CONCURRENT" \
        --eval-mode both
}

eval_baseline_no_thinking() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_file="results/${model_name}_baseline_no_thinking.json"
    local output_file="results/evaluated_${model_name}_baseline_no_thinking.json"

    if [[ -f "$output_file" ]]; then
        echo "Skipping eval baseline_no_thinking for $model (output exists: $output_file)"
        return 0
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file - skipping evaluation"
        return 1
    fi

    echo "============================================================"
    echo "Evaluating baseline_no_thinking for $model"
    echo "============================================================"
    python honesty_eval/evaluate.py \
        --attack-type baseline \
        --input "$input_file" \
        --output "$output_file" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --honesty-model "$HONESTY_MODEL" \
        --max-concurrent "$EVAL_CONCURRENT" \
        --eval-mode both
}

eval_system_prompt() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_dir="results/system_prompts/${model_name}"

    if [[ ! -d "$input_dir" ]]; then
        echo "Input directory not found: $input_dir - skipping evaluation"
        return 1
    fi

    for input_file in "$input_dir"/*.json; do
        if [[ ! -f "$input_file" ]]; then
            continue
        fi

        local base_name=$(basename "$input_file" .json)
        # Skip already evaluated files
        if [[ "$base_name" == evaluated_* ]]; then
            continue
        fi
        local output_file="${input_dir}/evaluated_${base_name}.json"
        if [[ -f "$output_file" ]]; then
            echo "Skipping eval system_prompt: $base_name (output exists)"
            continue
        fi

        echo "============================================================"
        echo "Evaluating system_prompt: $base_name"
        echo "============================================================"
        python honesty_eval/evaluate.py \
            --attack-type baseline \
            --input "$input_file" \
            --output "$output_file" \
            --facts "$FACTS_PATH" \
            --evaluator-model "$EVALUATOR_MODEL" \
            --honesty-model "$HONESTY_MODEL" \
            --max-concurrent "$EVAL_CONCURRENT" \
            --eval-mode both
    done
}

eval_assistant_prefill() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_file="results/${model_name}_assistant_prefill.json"
    local output_file="results/evaluated_${model_name}_assistant_prefill.json"

    if [[ -f "$output_file" ]]; then
        echo "Skipping eval assistant_prefill for $model (output exists: $output_file)"
        return 0
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file - skipping evaluation"
        return 1
    fi

    echo "============================================================"
    echo "Evaluating assistant_prefill for $model"
    echo "============================================================"
    python honesty_eval/evaluate.py \
        --attack-type assistant_prefill \
        --input "$input_file" \
        --output "$output_file" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --honesty-model "$HONESTY_MODEL" \
        --max-concurrent "$EVAL_CONCURRENT" \
        --eval-mode both
}

eval_user_prefill() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_file="results/${model_name}_user_prefill.json"
    local output_file="results/evaluated_${model_name}_user_prefill.json"

    if [[ -f "$output_file" ]]; then
        echo "Skipping eval user_prefill for $model (output exists: $output_file)"
        return 0
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file - skipping evaluation"
        return 1
    fi

    echo "============================================================"
    echo "Evaluating user_prefill for $model"
    echo "============================================================"
    python honesty_eval/evaluate.py \
        --attack-type user_prefill \
        --input "$input_file" \
        --output "$output_file" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --honesty-model "$HONESTY_MODEL" \
        --max-concurrent "$EVAL_CONCURRENT" \
        --eval-mode both
}

eval_user_prefill_simple() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_file="results/${model_name}_user_prefill_simple.json"
    local output_file="results/evaluated_${model_name}_user_prefill_simple.json"

    if [[ -f "$output_file" ]]; then
        echo "Skipping eval user_prefill_simple for $model (output exists: $output_file)"
        return 0
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file - skipping evaluation"
        return 1
    fi

    echo "============================================================"
    echo "Evaluating user_prefill_simple for $model"
    echo "============================================================"
    python honesty_eval/evaluate.py \
        --attack-type user_prefill_simple \
        --input "$input_file" \
        --output "$output_file" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --honesty-model "$HONESTY_MODEL" \
        --max-concurrent "$EVAL_CONCURRENT" \
        --eval-mode both
}

eval_pretrain_prompt() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_dir="results/pretrain_prompts/${model_name}"

    if [[ ! -d "$input_dir" ]]; then
        echo "Input directory not found: $input_dir - skipping evaluation"
        return 1
    fi

    for input_file in "$input_dir"/*.json; do
        if [[ ! -f "$input_file" ]]; then
            continue
        fi

        local base_name=$(basename "$input_file" .json)
        # Skip already evaluated files
        if [[ "$base_name" == evaluated_* ]]; then
            continue
        fi
        local output_file="${input_dir}/evaluated_${base_name}.json"
        if [[ -f "$output_file" ]]; then
            echo "Skipping eval pretrain_prompt: $base_name (output exists)"
            continue
        fi

        echo "============================================================"
        echo "Evaluating pretrain_prompt: $base_name"
        echo "============================================================"
        python honesty_eval/evaluate.py \
            --attack-type baseline \
            --input "$input_file" \
            --output "$output_file" \
            --facts "$FACTS_PATH" \
            --evaluator-model "$EVALUATOR_MODEL" \
            --honesty-model "$HONESTY_MODEL" \
            --max-concurrent "$EVAL_CONCURRENT" \
            --eval-mode both
    done
}

eval_pretrain() {
    local model=$1
    local model_name=$(echo "$model" | tr '/' '_' | tr '-' '_')
    local input_file="results/${model_name}_pretrain.json"
    local output_file="results/evaluated_${model_name}_pretrain.json"

    if [[ -f "$output_file" ]]; then
        echo "Skipping eval pretrain for $model (output exists: $output_file)"
        return 0
    fi

    if [[ ! -f "$input_file" ]]; then
        echo "Input file not found: $input_file - skipping evaluation"
        return 1
    fi

    echo "============================================================"
    echo "Evaluating pretrain for $model"
    echo "============================================================"
    python honesty_eval/evaluate.py \
        --attack-type pretrain \
        --input "$input_file" \
        --output "$output_file" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --honesty-model "$HONESTY_MODEL" \
        --max-concurrent "$EVAL_CONCURRENT" \
        --eval-mode both
}

# Run evaluations
echo ""
echo "============================================================"
echo "RUNNING EVALUATIONS"
echo "============================================================"
echo ""

for model in "${MODEL_LIST[@]}"; do
    echo ""
    echo "============================================================"
    echo "EVALUATING MODEL: $model"
    echo "============================================================"
    echo ""

    if [[ -z "$ATTACK" || "$ATTACK" == "baseline" ]]; then
        eval_baseline "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "baseline_no_thinking" ]]; then
        eval_baseline_no_thinking "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "system_prompt" ]]; then
        eval_system_prompt "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "assistant_prefill" ]]; then
        eval_assistant_prefill "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "user_prefill" ]]; then
        eval_user_prefill "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "user_prefill_simple" ]]; then
        eval_user_prefill_simple "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "pretrain_prompt" ]]; then
        eval_pretrain_prompt "$model"
    fi

    if [[ -z "$ATTACK" || "$ATTACK" == "pretrain" ]]; then
        eval_pretrain "$model"
    fi
done

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Results saved to results/"
echo "============================================================"
