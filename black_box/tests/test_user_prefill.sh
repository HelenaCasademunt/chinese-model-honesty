#!/bin/bash
# Test script for user prefill attack generation and evaluation
# Uses minimal samples to avoid overwriting real data

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

echo "=========================================="
echo "Testing User Prefill Attack + Evaluation"
echo "=========================================="

# Generate user prefill attack responses (2 samples per prefill, 1 initial sample)
echo ""
echo "Step 1: Generating user prefill attack responses..."
python black_box/user_prefill_attack.py \
    --output results/test_data/test_user_prefill_responses.json \
    --num-samples 2 \
    --num-initial-samples 1 \
    --max-concurrent-questions 2 \
    --concurrency 5 \
    --max-tokens 256 \
    --initial-max-tokens 512

# Check if generation succeeded
if [ ! -f results/test_data/test_user_prefill_responses.json ]; then
    echo "ERROR: User prefill generation failed - no output file created"
    exit 1
fi

echo ""
echo "Step 2: Evaluating user prefill responses..."
python black_box/eval/evaluate.py \
    --attack-type user_prefill \
    --input results/test_data/test_user_prefill_responses.json \
    --output results/test_data/test_evaluated_user_prefill_responses.json \
    --max-concurrent 5

# Check if evaluation succeeded
if [ ! -f results/test_data/test_evaluated_user_prefill_responses.json ]; then
    echo "ERROR: User prefill evaluation failed - no output file created"
    exit 1
fi

echo ""
echo "=========================================="
echo "User prefill test completed successfully!"
echo "Output files:"
echo "  - results/test_data/test_user_prefill_responses.json"
echo "  - results/test_data/test_evaluated_user_prefill_responses.json"
echo "=========================================="
