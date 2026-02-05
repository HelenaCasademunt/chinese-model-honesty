#!/bin/bash

source /root/.venv/bin/activate

cd /root/chinese-model-honesty

CONFIGS=(
    "honesty_training/input/goals_split_personality.yaml"
    "honesty_training/input/followup_split_personality.yaml"
    "honesty_training/input/followup_anthropic.yaml"
    "honesty_training/input/followup_qwen.yaml"
    "honesty_training/input/goals_anthropic.yaml"
    "honesty_training/input/goals_qwen.yaml"
    "honesty_training/input/mixed_anthropic.yaml"
    "honesty_training/input/mixed_qwen.yaml"
    "honesty_training/input/mixed_split_personality.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    python honesty_training/finetune_qwen3_32b.py "$config"
done

echo "All honesty finetunes complete!"


CURRENT_DATE=$(date +%Y%m%d)

# Standard split personality data
echo "=========================================="
echo "Training on standard split personality data"
echo "=========================================="
python split_personality/finetune_qwen3_split_personality.py \
    --data split_personality/data/split_personality_qwen3.jsonl \
    --output-dir /workspace/qwen3-32b-split-personality-${CURRENT_DATE} \
    --hf-repo-id hcasademunt/qwen3-32b-split-personality-${CURRENT_DATE}

# A-prompt split personality data
echo "=========================================="
echo "Training on a-prompt split personality data"
echo "=========================================="
python split_personality/finetune_qwen3_split_personality.py \
    --data split_personality/data/split_personality_a_prompt_qwen3.jsonl \
    --output-dir /workspace/qwen3-32b-split-personality-a-prompt-${CURRENT_DATE} \
    --hf-repo-id hcasademunt/qwen3-32b-split-personality-a-prompt-${CURRENT_DATE}

echo "=========================================="
echo "Both models trained and uploaded!"
echo "=========================================="
