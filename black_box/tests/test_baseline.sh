#!/bin/bash
# Test script for baseline generation and evaluation
# Uses minimal samples to avoid overwriting real data

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

echo "=========================================="
echo "Testing Baseline Generation + Evaluation"
echo "=========================================="

# Generate baseline responses (2 samples, limited questions)
echo ""
echo "Step 1: Generating baseline responses..."
python black_box/baseline.py \
    --output results/test_data/test_baseline_responses.json \
    --num-samples 2 \
    --max-concurrent 2 \
    --max-tokens 512

# Check if generation succeeded
if [ ! -f results/test_data/test_baseline_responses.json ]; then
    echo "ERROR: Baseline generation failed - no output file created"
    exit 1
fi

echo ""
echo "Step 2: Evaluating baseline responses..."
python black_box/eval/evaluate.py \
    --attack-type baseline \
    --input results/test_data/test_baseline_responses.json \
    --output results/test_data/test_evaluated_baseline_responses.json \
    --max-concurrent 5

# Check if evaluation succeeded
if [ ! -f results/test_data/test_evaluated_baseline_responses.json ]; then
    echo "ERROR: Baseline evaluation failed - no output file created"
    exit 1
fi

echo ""
echo "=========================================="
echo "Baseline test completed successfully!"
echo "Output files:"
echo "  - results/test_data/test_baseline_responses.json"
echo "  - results/test_data/test_evaluated_baseline_responses.json"
echo "=========================================="
