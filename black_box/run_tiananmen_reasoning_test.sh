#!/bin/bash

source /root/.venv/bin/activate

MODEL="qwen/qwen3-32b"
INPUT_DATA="/root/chinese_auditing/data/tiananmen_square_1989.json"
QUESTIONS="black_box/eval/prompts/tiananmen_questions.json"
FACTS="black_box/eval/prompts/tiananmen_facts.json"
SAMPLES=3
CONCURRENCY=10
OUTDIR="results/data/new_pipeline_reasoning_test"

mkdir -p "$OUTDIR"

# Convert source data to questions and facts format
echo "============================================"
echo "Converting tiananmen data"
echo "============================================"
python black_box/convert_tiananmen.py \
    --input "$INPUT_DATA" \
    --questions-output "$QUESTIONS" \
    --facts-output "$FACTS"

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

# Evaluate all results
echo ""
echo "============================================"
echo "Evaluating results"
echo "============================================"
for EFFORT in default high low none; do
    INPUT="${OUTDIR}/baseline_responses_reasoning_${EFFORT}.json"
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
        --facts "$FACTS" \
        --max-concurrent "$CONCURRENCY"
done
