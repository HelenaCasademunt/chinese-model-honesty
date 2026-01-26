#!/bin/bash
# Run assistant prefill attack with finegrained filtered prefills and evaluate

cd "$(dirname "$0")/.."

# Step 1: Run attack
python black_box/assistant_prefill_attack.py \
    --questions black_box/eval/prompts/finegrained_eval_questions_filtered.json \
    --custom-prefills black_box/prompts/finegrained_assistant_prefill_filtered.json \
    --output results/data/finegrained_assistant_prefill_responses.json \
    --mode skip

# Step 2: Evaluate
python black_box/eval/evaluate_openrouter.py \
    --attack-type assistant_prefill \
    --input results/data/finegrained_assistant_prefill_responses.json \
    --output results/data/evaluated_finegrained_assistant_prefill_responses.json \
    --facts black_box/eval/prompts/finegrained_eval_facts_filtered.json
