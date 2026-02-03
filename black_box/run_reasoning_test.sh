#!/bin/bash

source /root/.venv/bin/activate

MODEL="qwen/qwen3-32b"
QUESTIONS="black_box/eval/prompts/finegrained_eval_questions_filtered.json"
SAMPLES=3
CONCURRENCY=10
OUTDIR="results/data/reasoning_test"

mkdir -p "$OUTDIR"

# Default (no reasoning effort specified, model decides)
echo "============================================"
echo "Running with default reasoning (no flag)"
echo "============================================"
python black_box/baseline.py \
    --model "$MODEL" \
    --questions "$QUESTIONS" \
    --output "${OUTDIR}/baseline_responses_reasoning_default.json" \
    --num-samples "$SAMPLES" \
    --max-concurrent "$CONCURRENCY"

for EFFORT in high low none; do
    echo "============================================"
    echo "Running with reasoning effort: $EFFORT"
    echo "============================================"
    python black_box/baseline.py \
        --model "$MODEL" \
        --questions "$QUESTIONS" \
        --output "${OUTDIR}/baseline_responses_reasoning_${EFFORT}.json" \
        --num-samples "$SAMPLES" \
        --max-concurrent "$CONCURRENCY" \
        --reasoning-effort "$EFFORT"
done
