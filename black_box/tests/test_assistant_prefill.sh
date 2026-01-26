#!/bin/bash
# Test script for assistant prefill attack generation and evaluation
# Uses minimal samples to avoid overwriting real data

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

echo "=========================================="
echo "Testing Assistant Prefill Attack + Evaluation"
echo "=========================================="

# Generate assistant prefill attack responses (2 samples)
echo ""
echo "Step 1: Generating assistant prefill attack responses..."
python black_box/assistant_prefill_attack.py \
    --output results/test_data/test_assistant_prefill_responses.json \
    --custom-prefills black_box/prompts/finegrained_assistant_prefill.json \
    --num-samples 2 \
    --max-concurrent-questions 2 \
    --concurrency 5 \
    --max-tokens 512

# Check if generation succeeded
if [ ! -f results/test_data/test_assistant_prefill_responses.json ]; then
    echo "ERROR: Assistant prefill generation failed - no output file created"
    exit 1
fi

echo ""
echo "Step 2: Evaluating assistant prefill responses..."
python black_box/eval/evaluate.py \
    --attack-type assistant_prefill \
    --input results/test_data/test_assistant_prefill_responses.json \
    --output results/test_data/test_evaluated_assistant_prefill_responses.json \
    --max-concurrent 5

# Check if evaluation succeeded
if [ ! -f results/test_data/test_evaluated_assistant_prefill_responses.json ]; then
    echo "ERROR: Assistant prefill evaluation failed - no output file created"
    exit 1
fi

echo ""
echo "=========================================="
echo "Assistant prefill test completed successfully!"
echo "Output files:"
echo "  - results/test_data/test_assistant_prefill_responses.json"
echo "  - results/test_data/test_evaluated_assistant_prefill_responses.json"
echo "=========================================="
