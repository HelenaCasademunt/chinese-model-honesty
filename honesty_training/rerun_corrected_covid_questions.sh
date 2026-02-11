#!/bin/bash

# Re-generate responses for the 2 COVID questions whose wording was updated
# to include "China's" (matching dev_facts_explicit.json in chinese_auditing).
# Then merge the new responses into the original result files.

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
QUESTIONS="honesty_training/corrected_covid_questions.json"
RESULTS_DIR="results/honesty"
TEMP_DIR="results/honesty/temp_covid_rerun"
mkdir -p "$TEMP_DIR"

# Same configs as run_all_deepseek_evaluations.sh
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

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)
    OUTPUT_DIR=$(grep "^output_dir:" "$config" | awk '{print $2}')
    LORA_NAME="${config_name#deepseek_}"

    ORIGINAL="$RESULTS_DIR/deepseek-r1-70b-${LORA_NAME}.json"
    TEMP_OUTPUT="$TEMP_DIR/deepseek-r1-70b-${LORA_NAME}.json"

    if [ ! -f "$ORIGINAL" ]; then
        echo "SKIP: $ORIGINAL does not exist"
        continue
    fi

    echo "=========================================="
    echo "Re-running COVID questions: $config_name"
    echo "LoRA: $OUTPUT_DIR"
    echo "Started at: $(date)"
    echo "=========================================="

    python honesty_eval/sample_assistant_responses_local.py \
        --model "$BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$TEMP_OUTPUT" \
        --temperature 0.7 \
        --num-samples 10 \
        --batch-size 200 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora "$OUTPUT_DIR" \
        --mode overwrite

    if [ $? -eq 0 ]; then
        echo "Merging into $ORIGINAL..."
        python honesty_training/merge_corrected_responses.py "$ORIGINAL" "$TEMP_OUTPUT"
        echo "Done: $config_name"
    else
        echo "FAILED: $config_name"
    fi

    echo ""
done

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo "=========================================="
echo "All corrected COVID question runs completed at: $(date)"
echo "=========================================="
