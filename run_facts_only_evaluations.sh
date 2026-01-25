#!/bin/bash

# Run all facts-only evaluations
# These scripts use existing refusal classifications and only evaluate facts

source /root/.venv/bin/activate

FACTS_FILE="black_box/eval/prompts/finegrained_eval_facts.json"

echo "========================================"
echo "Running Facts-Only Evaluations"
echo "========================================"

# # Baseline facts only
# echo ""
# echo "1. Evaluating baseline responses (facts only)..."
# python black_box/eval/evaluate_facts_only.py \
#     --input results/data/evaluated_responses.json \
#     --output results/data/evaluated_responses_facts_only.json \
#     --facts "$FACTS_FILE"

# # Baseline facts only (llama)
# if [ -f "results/data/evaluated_responses_llama70b.json" ]; then
#     echo ""
#     echo "2. Evaluating baseline responses - llama (facts only)..."
#     python black_box/eval/evaluate_facts_only.py \
#         --input results/data/evaluated_responses_llama70b.json \
#         --output results/data/evaluated_responses_facts_only_llama70b.json \
#         --facts "$FACTS_FILE"
# fi

# # Pretrain attack facts only
# if [ -f "results/data/evaluated_pretrain_attack_responses.json" ]; then
#     echo ""
#     echo "3. Evaluating pretrain attack responses (facts only)..."
#     python black_box/eval/evaluate_pretrain_facts_only.py \
#         --input results/data/evaluated_pretrain_attack_responses.json \
#         --output results/data/evaluated_pretrain_attack_responses_facts_only.json \
#         --facts "$FACTS_FILE"
# fi

# User prefill facts only
echo ""
echo "4. Evaluating user prefill responses (facts only)..."
python black_box/eval/evaluate_user_prefill_facts_only.py \
    --max-concurrent 50

# Standard assistant prefill facts only
if [ -f "results/data/evaluated_standard_assistant_prefill_responses.json" ]; then
    echo ""
    echo "5. Evaluating standard assistant prefill responses (facts only)..."
    python black_box/eval/evaluate_assistant_prefill_facts_only.py \
        --input results/data/evaluated_standard_assistant_prefill_responses.json \
        --output results/data/evaluated_standard_assistant_prefill_responses_facts_only.json \
        --facts "$FACTS_FILE" \
        --max-concurrent 50
fi

# Finegrained assistant prefill facts only
if [ -f "results/data/evaluated_finegrained_assistant_prefill_responses.json" ]; then
    echo ""
    echo "6. Evaluating finegrained assistant prefill responses (facts only)..."
    python black_box/eval/evaluate_assistant_prefill_facts_only.py \
        --input results/data/evaluated_finegrained_assistant_prefill_responses.json \
        --output results/data/evaluated_finegrained_assistant_prefill_responses_facts_only.json \
        --facts "$FACTS_FILE" \
        --max-concurrent 50

fi

echo ""
echo "========================================"
echo "Facts-Only Evaluations Complete!"
echo "========================================"
