#!/bin/bash

source /root/.venv/bin/activate

MODEL="deepseek/deepseek-r1-distill-llama-70b"
GOAL_INPUT="/root/chinese_auditing/src/honesty_finetuning/data/goal_data_honest_original_chat.jsonl"
FOLLOWUP_INPUT="/root/chinese_auditing/src/honesty_finetuning/data/followup_data_original_chat.jsonl"
OUTPUT_DIR="honesty_training/data"

echo "=== Collecting goal + followup responses with DeepSeek (in parallel) ==="

python honesty_training/collect_goal_responses.py \
    --goals "$GOAL_INPUT" \
    --source chat \
    --model "$MODEL" \
    --output "$OUTPUT_DIR/deepseek_goal_responses.json" \
    --num-samples 1 \
    --max-concurrent 50 &
PID_GOAL=$!

python honesty_training/collect_followup_responses.py \
    --input "$FOLLOWUP_INPUT" \
    --source chat \
    --model "$MODEL" \
    --output "$OUTPUT_DIR/deepseek_followup_responses.json" \
    --num-samples 1 \
    --max-concurrent 50 &
PID_FOLLOWUP=$!

echo "Goal PID: $PID_GOAL, Followup PID: $PID_FOLLOWUP"
wait $PID_GOAL
echo "=== Goal collection finished ==="
wait $PID_FOLLOWUP
echo "=== Followup collection finished ==="

echo ""
echo "=== Converting to chat format ==="
python honesty_training/format_deepseek_chat.py \
    --goal-input "$OUTPUT_DIR/deepseek_goal_responses.json" \
    --goal-output "$OUTPUT_DIR/goal_data_deepseek_chat.jsonl" \
    --followup-input "$OUTPUT_DIR/deepseek_followup_responses.json" \
    --followup-output "$OUTPUT_DIR/followup_data_deepseek_chat.jsonl"

echo ""
echo "=== Done ==="
echo "Goal data:     $OUTPUT_DIR/goal_data_deepseek_chat.jsonl"
echo "Followup data: $OUTPUT_DIR/followup_data_deepseek_chat.jsonl"
