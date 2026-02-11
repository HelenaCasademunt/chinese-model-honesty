#!/bin/bash

# Activate virtual environment
source /root/.venv/bin/activate

# Change to project directory
cd /root/chinese-model-honesty

# Ensure /tmp directory exists and is writable
if [ ! -d "/tmp" ]; then
    mkdir -p /tmp
fi
export TMPDIR=/tmp

# Base model
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# Questions file
QUESTIONS="data/dev_questions.json"

# Array of config files to run
configs=(
    "honesty_training/input/deepseek_goals_qwen.yaml"
    "honesty_training/input/deepseek_goals_anthropic.yaml"
    "honesty_training/input/deepseek_goals_split_personality.yaml"
    "honesty_training/input/deepseek_followup_qwen.yaml"
    "honesty_training/input/deepseek_followup_anthropic.yaml"
    "honesty_training/input/deepseek_followup_split_personality.yaml"
    "honesty_training/input/deepseek_mixed_qwen.yaml"
    "honesty_training/input/deepseek_mixed_anthropic.yaml"
    "honesty_training/input/deepseek_mixed_split_personality.yaml"
    "honesty_training/input/deepseek_control_chinese_topics.yaml"
    "honesty_training/input/deepseek_control_alpaca.yaml"
    "honesty_training/input/deepseek_control_openhermes.yaml"
)

# Results directory
RESULTS_DIR="results/honesty"
mkdir -p "$RESULTS_DIR"

# Run evaluation for each config
for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)

    # Extract output_dir from config file
    OUTPUT_DIR=$(grep "^output_dir:" "$config" | awk '{print $2}')

    # Generate lora name (remove deepseek_ prefix for cleaner names)
    LORA_NAME="${config_name#deepseek_}"

    echo "=========================================="
    echo "Evaluating finetuned model: $config_name"
    echo "Output dir: $OUTPUT_DIR"
    echo "Started at: $(date)"
    echo "=========================================="

    python honesty_eval/sample_assistant_responses_local.py \
        --model "$BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$RESULTS_DIR/deepseek-r1-70b-${LORA_NAME}.json" \
        --temperature 0.7 \
        --num-samples 10 \
        --batch-size 200 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora "$OUTPUT_DIR"

    eval_exit_code=$?

    if [ $eval_exit_code -eq 0 ]; then
        echo "✓ Successfully evaluated: $config_name"
        echo "Results saved to: $RESULTS_DIR/deepseek-r1-70b-${LORA_NAME}.json"
    else
        echo "✗ Evaluation failed: $config_name (exit code: $eval_exit_code)"
    fi

    echo ""
done

echo "=========================================="
echo "All evaluation runs completed at: $(date)"
echo "Results saved to: $RESULTS_DIR/deepseek-r1-70b-*.json"
echo "=========================================="
