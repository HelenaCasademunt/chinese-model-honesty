#!/bin/bash

# Run lie detection evaluations for all honesty finetuned models
# This script runs:
# 1. ask_if_true_confession_balanced_local.py - asks model if it lied (conversation format)
# 2. ask_if_true_balanced_local.py - asks model to evaluate truthfulness
# 3. evaluate_confession.py - classifies confession responses using GPT

source /root/.venv/bin/activate

BASE_MODEL="Qwen/Qwen3-32B"
INPUT_FILE="results/evaluated_qwen_qwen3_32b_baseline_no_thinking.json"
OUTPUT_DIR="results/honesty_lie_detection"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# "qwen3-32b-control-alpaca"
# "qwen3-32b-lora-finetuned-followup-anthropic"
# "qwen3-32b-lora-finetuned-followup-qwen-data"
# "qwen3-32b-lora-finetuned-mixed-split-personality"
# "qwen3-32b-lora-finetuned-followup-split-personality-data"

# All LoRA adapters to evaluate
ADAPTERS=(
    "qwen3-32b-lora-finetuned-goals-anthropic"
    "qwen3-32b-lora-finetuned-goals-qwen"
    "qwen3-32b-lora-finetuned-honesty-split-personality"
    "qwen3-32b-lora-finetuned-mixed-anthropic"
    "qwen3-32b-lora-finetuned-mixed-qwen"
    "qwen3-32b-split-personality-20260205"
    "qwen3-32b-split-personality-a-prompt-20260205"
    "qwen3-32b-control-openhermes"
)

for adapter in "${ADAPTERS[@]}"; do
    echo "=============================================="
    echo "Processing adapter: $adapter"
    echo "=============================================="

    adapter_path="/workspace/$adapter"

    # Output file names
    confession_output="$OUTPUT_DIR/${adapter}_confession.json"
    truthfulness_output="$OUTPUT_DIR/${adapter}_truthfulness.json"
    confession_eval_output="$OUTPUT_DIR/${adapter}_confession_evaluated.json"

    # Skip if already completed
    if [[ -f "$confession_eval_output" ]]; then
        echo "Skipping $adapter - already completed"
        continue
    fi

    # 1. Run confession asking (conversation format)
    echo "Running confession asking..."
    python honesty_training/ask_if_true_confession_balanced_local.py \
        --model "$BASE_MODEL" \
        --input "$INPUT_FILE" \
        --output "$confession_output" \
        --lora-adapter "$adapter_path" \
        --tensor-parallel-size 1 \
        --batch-size 200 \
        --disable-compile \
        --temperature 0.7 \
        --max-tokens 2048

    # 2. Run truthfulness evaluation
    echo "Running truthfulness evaluation..."
    python honesty_training/ask_if_true_balanced_local.py \
        --model "$BASE_MODEL" \
        --input "$INPUT_FILE" \
        --output "$truthfulness_output" \
        --lora-adapter "$adapter_path" \
        --tensor-parallel-size 1 \
        --batch-size 200 \
        --disable-compile \
        --temperature 0.7 \
        --max-tokens 1024

    # 3. Evaluate confessions with GPT
    echo "Evaluating confessions with GPT..."
    python asking_the_model/evaluate_confession.py \
        --input "$confession_output" \
        --output "$confession_eval_output" \
        --evaluator-model "gpt-5.2" \
        --max-concurrent 20

    echo "Completed $adapter"
    echo ""
done

echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
