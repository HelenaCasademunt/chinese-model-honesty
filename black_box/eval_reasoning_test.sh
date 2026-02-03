#!/bin/bash

source /root/.venv/bin/activate

INDIR="results/data/reasoning_test"
OUTDIR="results/data/reasoning_test"
CONCURRENCY=10

for EFFORT in default high low none; do
    INPUT="${INDIR}/baseline_responses_reasoning_${EFFORT}.json"
    OUTPUT="${OUTDIR}/eval_responses_reasoning_${EFFORT}.json"

    if [ ! -f "$INPUT" ]; then
        echo "Skipping $EFFORT: $INPUT not found"
        continue
    fi

    echo "============================================"
    echo "Evaluating reasoning effort: $EFFORT"
    echo "============================================"
    python black_box/eval/evaluate_openrouter.py \
        --attack-type baseline \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --max-concurrent "$CONCURRENCY"
done
