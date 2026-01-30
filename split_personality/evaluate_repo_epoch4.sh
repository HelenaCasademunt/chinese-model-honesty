#!/bin/bash

# Evaluate split-personality-repo-epoch-4 model
# 1. Generate assistant responses with empty system prompt (to match intervention template)
# 2. Evaluate with split personality evaluation

# Configuration
BASE_MODEL="Qwen/Qwen3-32B"
LORA_ADAPTER="hcasademunt/split-personality-repo-epoch-4"
QUESTIONS_FILE="black_box/eval/prompts/finegrained_eval_questions_filtered.json"
FACTS_FILE="black_box/eval/prompts/finegrained_eval_facts_filtered.json"
RESULTS_DIR="split_personality/results"

# Output files
ASSISTANT_OUTPUT="${RESULTS_DIR}/repo_epoch4_assistant_only.json"
EVAL_OUTPUT="${RESULTS_DIR}/repo_epoch4_assistant_only_eval.json"

# GPU settings
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY=0.95
MAX_MODEL_LEN=8192
BATCH_SIZE=8
NUM_SAMPLES=10

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Evaluating split-personality-repo-epoch-4"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "LoRA adapter: $LORA_ADAPTER"
echo "Questions: $QUESTIONS_FILE"
echo "Output: $ASSISTANT_OUTPUT"
echo "=========================================="
echo ""

# Step 1: Generate assistant-only responses with EMPTY system prompt
# This matches the template used in sample_honest_persona_tokenized.py
echo "Step 1: Generating assistant-only responses..."
echo "Using empty system prompt to match intervention template"
echo ""

python split_personality/generate_responses_assistant_only.py \
    --model "$BASE_MODEL" \
    --lora-adapter "$LORA_ADAPTER" \
    --questions "$QUESTIONS_FILE" \
    --output "$ASSISTANT_OUTPUT" \
    --system-prompt "" \
    --temperature 0.7 \
    --num-samples $NUM_SAMPLES \
    --max-tokens 3072 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len $MAX_MODEL_LEN \
    --batch-size $BATCH_SIZE \
    --disable-compile \
    --template default \
    --mode skip

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate assistant responses"
    exit 1
fi

echo ""
echo "Step 1 complete: Responses saved to $ASSISTANT_OUTPUT"
echo ""

# Step 2: Evaluate assistant responses with split personality evaluation
echo "=========================================="
echo "Step 2: Evaluating assistant responses..."
echo ""

python split_personality/evaluate_split_personality.py \
    --input "$ASSISTANT_OUTPUT" \
    --output "$EVAL_OUTPUT" \
    --facts "$FACTS_FILE" \
    --evaluator-model "google/gemini-3-flash-preview" \
    --max-concurrent 50

if [ $? -ne 0 ]; then
    echo "Error: Failed to evaluate assistant responses"
    exit 1
fi

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo "Assistant responses: $ASSISTANT_OUTPUT"
echo "Evaluation results: $EVAL_OUTPUT"
echo "=========================================="
