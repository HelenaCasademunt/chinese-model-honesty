#!/bin/bash
# Test script for pretrain attack generation and evaluation
# Uses minimal samples to avoid overwriting real data
# NOTE: Requires baseline responses to exist first (run test_baseline.sh first)

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

echo "=========================================="
echo "Testing Pretrain Attack + Evaluation"
echo "=========================================="

# Check if baseline responses exist
if [ ! -f results/test_data/test_baseline_responses.json ]; then
    echo "ERROR: Baseline responses not found at results/test_data/test_baseline_responses.json"
    echo "Please run test_baseline.sh first to generate baseline responses"
    exit 1
fi

# Generate pretrain attack responses (2 samples)
echo ""
echo "Step 1: Generating pretrain attack responses..."
python black_box/pretrain_attack.py \
    --baseline results/test_data/test_baseline_responses.json \
    --output results/test_data/test_pretrain_responses.json \
    --num-samples 2 \
    --max-concurrent-questions 2 \
    --concurrency 5 \
    --max-tokens 512

# Check if generation succeeded
if [ ! -f results/test_data/test_pretrain_responses.json ]; then
    echo "ERROR: Pretrain generation failed - no output file created"
    exit 1
fi

echo ""
echo "Step 2: Evaluating pretrain responses..."
python black_box/eval/evaluate.py \
    --attack-type pretrain \
    --input results/test_data/test_pretrain_responses.json \
    --output results/test_data/test_evaluated_pretrain_responses.json \
    --max-concurrent 5

# Check if evaluation succeeded
if [ ! -f results/test_data/test_evaluated_pretrain_responses.json ]; then
    echo "ERROR: Pretrain evaluation failed - no output file created"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pretrain test completed successfully!"
echo "Output files:"
echo "  - results/test_data/test_pretrain_responses.json"
echo "  - results/test_data/test_evaluated_pretrain_responses.json"
echo "=========================================="
