#!/bin/bash

source /root/.venv/bin/activate

SRC_DIR="/root/chinese_auditing/output/prompt_tests"
CONVERTED_DIR="black_box/eval/prompts/prompt_tests_converted"
OUTDIR="results/data/prompt_tests_no_thinking"
MODEL="qwen/qwen3-32b"
SAMPLES=3
CONCURRENCY=30

mkdir -p "$CONVERTED_DIR"
mkdir -p "$OUTDIR"

# Convert all source files
for src_file in "$SRC_DIR"/*/*/*.json; do
    # Extract model_name/topic/variant.json
    rel="${src_file#$SRC_DIR/}"
    model_name=$(echo "$rel" | cut -d/ -f1)
    topic=$(echo "$rel" | cut -d/ -f2)
    variant=$(basename "$rel" .json)

    out_file="${CONVERTED_DIR}/${model_name}/${topic}/${variant}.json"
    mkdir -p "$(dirname "$out_file")"
    python black_box/convert_format.py "$src_file" "$out_file" --topic "$topic"
done

echo "Conversion complete."

# Run baseline_no_thinking.py on each converted file
for conv_file in "$CONVERTED_DIR"/*/*/*.json; do
    rel="${conv_file#$CONVERTED_DIR/}"
    model_name=$(echo "$rel" | cut -d/ -f1)
    topic=$(echo "$rel" | cut -d/ -f2)
    variant=$(basename "$rel" .json)

    result_file="${OUTDIR}/${model_name}/${topic}/${variant}.json"
    mkdir -p "$(dirname "$result_file")"

    echo "============================================"
    echo "Running: model_name=$model_name topic=$topic variant=$variant"
    echo "============================================"
    python black_box/baseline_no_thinking.py \
        --model "$MODEL" \
        --questions "$conv_file" \
        --output "$result_file" \
        --num-samples "$SAMPLES" \
        --max-concurrent "$CONCURRENCY"
done
