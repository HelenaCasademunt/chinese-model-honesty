#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing

CONFIGS=(
    configs/honesty_finetuning/eval_deepseek_control_alpaca.yaml
    configs/honesty_finetuning/eval_deepseek_control_chinese_topics.yaml
    configs/honesty_finetuning/eval_deepseek_control_openhermes.yaml
    configs/honesty_finetuning/eval_deepseek_followup_anthropic.yaml
    configs/honesty_finetuning/eval_deepseek_followup_qwen.yaml
    configs/honesty_finetuning/eval_deepseek_followup_split_personality.yaml
    configs/honesty_finetuning/eval_deepseek_goals_anthropic.yaml
    configs/honesty_finetuning/eval_deepseek_goals_qwen.yaml
    configs/honesty_finetuning/eval_deepseek_goals_split_personality.yaml
    configs/honesty_finetuning/eval_deepseek_mixed_anthropic.yaml
    configs/honesty_finetuning/eval_deepseek_mixed_qwen.yaml
    configs/honesty_finetuning/eval_deepseek_mixed_split_personality.yaml
)

TOTAL=${#CONFIGS[@]}
PASSED=0
FAILED=0

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NUM=$((i + 1))
    echo "======================================"
    echo "[$NUM/$TOTAL] Running: $CONFIG"
    echo "======================================"

    python src/evaluation/evaluation_pipeline.py "$CONFIG"

    if [ $? -eq 0 ]; then
        echo "[$NUM/$TOTAL] PASSED: $CONFIG"
        PASSED=$((PASSED + 1))
    else
        echo "[$NUM/$TOTAL] FAILED: $CONFIG"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "======================================"
echo "Summary: $PASSED/$TOTAL passed, $FAILED failed"
echo "======================================"
