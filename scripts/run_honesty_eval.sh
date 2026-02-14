#!/bin/bash

source /root/.venv/bin/activate

INPUT_DIR="results/honesty"
OUTPUT_DIR="results/honesty/evaluated"

mkdir -p "$OUTPUT_DIR"

for input_file in "$INPUT_DIR"/*.json; do
    filename=$(basename "$input_file")
    output_file="$OUTPUT_DIR/${filename%.json}_evaluated.json"

    echo "=========================================="
    echo "Evaluating: $filename"
    echo "Output: $output_file"
    echo "=========================================="

    python -m honesty_eval.evaluate \
        --attack-type baseline \
        --input "$input_file" \
        --output "$output_file" \
        --eval-mode both \
        --honesty-prompt-version 1

    echo ""
done

echo "All evaluations complete!"
