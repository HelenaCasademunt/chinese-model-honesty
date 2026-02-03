#!/usr/bin/env bash
# Run baseline_v2.py on all prompt_tests input files.
# Output mirrors the input directory structure under results/data/baseline_v2/

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

INPUT_ROOT="/root/chinese_auditing/output/prompt_tests"
OUTPUT_ROOT="/root/chinese-model-honesty/results/data/baseline_v2"
MODEL="${1:-qwen/qwen3-32b}"
NUM_SAMPLES="${2:-10}"

echo "Model: $MODEL"
echo "Samples per question: $NUM_SAMPLES"
echo "Input root: $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo ""

for input_file in $(find "$INPUT_ROOT" -type f -name "*.json" | sort); do
    # Extract relative path: e.g. gemini-3-flash-preview/tiananmen_square_1989/baseline.json
    rel_path="${input_file#$INPUT_ROOT/}"
    # Build output path
    output_file="$OUTPUT_ROOT/$rel_path"

    echo "========================================"
    echo "INPUT:  $input_file"
    echo "OUTPUT: $output_file"
    echo "========================================"

    python black_box_v2/baseline_v2.py \
        --questions "$input_file" \
        --output "$output_file" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --max-concurrent 10 \
        --max-tokens 3072 \
        --temperature 0.7 \
        --mode skip

    echo ""
done

echo "All done!"
