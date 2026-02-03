#!/bin/bash
# Evaluate all prompt test responses using evaluate_openrouter.py.
# Expects converted fact files in black_box/eval/prompts/prompt_tests_converted/
# and response files in results/data/prompt_tests/

source /root/.venv/bin/activate
cd /root/chinese-model-honesty

RESPONSES_DIR="results/data/prompt_tests"
FACTS_DIR="black_box/eval/prompts/prompt_tests_converted"
OUTPUT_DIR="results/data/prompt_tests_evaluated"

mkdir -p "$OUTPUT_DIR"

for model_dir in "$RESPONSES_DIR"/*/; do
    model=$(basename "$model_dir")

    for topic_dir in "$model_dir"*/; do
        topic=$(basename "$topic_dir")

        for response_file in "$topic_dir"*.json; do
            [ -f "$response_file" ] || continue
            variant=$(basename "$response_file" .json)

            facts_file="$FACTS_DIR/$model/$topic/$variant.json"
            if [ ! -f "$facts_file" ]; then
                echo "SKIP (no facts): $model/$topic/$variant"
                continue
            fi

            out_dir="$OUTPUT_DIR/$model/$topic"
            mkdir -p "$out_dir"
            output_file="$out_dir/$variant.json"

            if [ -f "$output_file" ]; then
                echo "SKIP (already evaluated): $model/$topic/$variant"
                continue
            fi

            echo "============================================================"
            echo "Evaluating: model=$model topic=$topic variant=$variant"
            echo "  Input:  $response_file"
            echo "  Facts:  $facts_file"
            echo "  Output: $output_file"
            echo "============================================================"

            python3 black_box/eval/evaluate_openrouter.py \
                --attack-type baseline \
                --input "$response_file" \
                --output "$output_file" \
                --facts "$facts_file"

            echo ""
            echo "Finished: $model/$topic/$variant"
            echo ""
        done
    done
done

echo "All prompt test evaluations complete."
