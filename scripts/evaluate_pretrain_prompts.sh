#!/bin/bash
# Evaluate all pretrain prompt attack response files

source /root/.venv/bin/activate

INPUT_DIR="results/data/pretrain_prompts"
OUTPUT_DIR="results/data/pretrain_prompts"
FACTS_PATH="black_box/eval/prompts/finegrained_eval_facts_filtered.json"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-google/gemini-3-flash-preview}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    echo "Run pretrain_prompt_attack.py first to generate responses"
    exit 1
fi

# Find all pretrain response files
FILES=$(find "$INPUT_DIR" -name "pretrain_*_responses.json" -type f)

if [ -z "$FILES" ]; then
    echo "No pretrain response files found in $INPUT_DIR"
    exit 1
fi

echo "============================================================"
echo "Evaluating pretrain prompt attack responses"
echo "============================================================"
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Evaluator model:  $EVALUATOR_MODEL"
echo "Max concurrent:   $MAX_CONCURRENT"
echo ""

# Process each file
for INPUT_FILE in $FILES; do
    BASENAME=$(basename "$INPUT_FILE")
    # Replace "pretrain_" prefix with "evaluated_pretrain_"
    OUTPUT_FILE="$OUTPUT_DIR/evaluated_${BASENAME}"

    echo "============================================================"
    echo "Processing: $BASENAME"
    echo "Output:     $(basename "$OUTPUT_FILE")"
    echo "============================================================"

    python black_box/eval/evaluate_openrouter.py \
        --attack-type pretrain \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_FILE" \
        --facts "$FACTS_PATH" \
        --evaluator-model "$EVALUATOR_MODEL" \
        --max-concurrent "$MAX_CONCURRENT"

    if [ $? -eq 0 ]; then
        echo "Successfully evaluated: $BASENAME"
    else
        echo "Error evaluating: $BASENAME"
    fi
    echo ""
done

echo "============================================================"
echo "All evaluations complete"
echo "============================================================"
