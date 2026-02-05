#!/bin/bash

source /root/.venv/bin/activate

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
