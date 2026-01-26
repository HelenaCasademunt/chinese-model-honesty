#!/bin/bash
# Run all evaluation pipeline tests
# Tests baseline, user prefill, assistant prefill, and pretrain attacks

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

echo "=========================================="
echo "Running All Evaluation Pipeline Tests"
echo "=========================================="
echo ""

# Create test data directory
mkdir -p results/test_data

# Track failures
FAILED_TESTS=""

# Test 1: Baseline
echo ">>> Running Baseline Test..."
bash black_box/tests/test_baseline.sh
if [ $? -ne 0 ]; then
    FAILED_TESTS="$FAILED_TESTS baseline"
    echo "!!! Baseline test FAILED"
else
    echo ">>> Baseline test PASSED"
fi
echo ""

# Test 2: User Prefill
echo ">>> Running User Prefill Test..."
bash black_box/tests/test_user_prefill.sh
if [ $? -ne 0 ]; then
    FAILED_TESTS="$FAILED_TESTS user_prefill"
    echo "!!! User Prefill test FAILED"
else
    echo ">>> User Prefill test PASSED"
fi
echo ""

# Test 3: Assistant Prefill
echo ">>> Running Assistant Prefill Test..."
bash black_box/tests/test_assistant_prefill.sh
if [ $? -ne 0 ]; then
    FAILED_TESTS="$FAILED_TESTS assistant_prefill"
    echo "!!! Assistant Prefill test FAILED"
else
    echo ">>> Assistant Prefill test PASSED"
fi
echo ""

# Test 4: Pretrain (depends on baseline)
echo ">>> Running Pretrain Test..."
bash black_box/tests/test_pretrain.sh
if [ $? -ne 0 ]; then
    FAILED_TESTS="$FAILED_TESTS pretrain"
    echo "!!! Pretrain test FAILED"
else
    echo ">>> Pretrain test PASSED"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ -z "$FAILED_TESTS" ]; then
    echo "All tests PASSED!"
    exit 0
else
    echo "FAILED tests:$FAILED_TESTS"
    exit 1
fi
