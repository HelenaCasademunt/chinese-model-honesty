#!/bin/bash
# Run confession and classification evaluations for all Qwen3-VL-8B honesty LoRA adapters

source /root/.venv/bin/activate

cd /root/chinese_auditing/src/confession

BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"
RESPONSES_DIR="/root/chinese_auditing/results/qwen3-vl-8b-thinking/honesty"
OUTPUT_BASE="/root/chinese_auditing/results/confession/qwen3-vl-8b-thinking/honesty"

# vLLM parameters
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=3072
MAX_TOKENS_CLASSIFICATION=1024
BATCH_SIZE=100
GPU_MEMORY=0.95
MAX_MODEL_LEN=8192

config_names=(
    "goals_qwen_vl_8b_thinking"
    "followup_qwen_vl_8b_thinking"
    "goals_anthropic"
    "followup_anthropic"
    "followup_split_personality"
    "split_personality_b_pass"
    "control_alpaca"
    "control_chinese_topics"
    "control_openhermes"
    "mixed_qwen_vl_8b_thinking"
)

TOTAL=${#config_names[@]}
FAILED=()

echo ""
echo "=========================================="
echo "Confession/Classification Evaluations for Honesty LoRA Adapters"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "Configs: $TOTAL"
echo ""

for i in "${!config_names[@]}"; do
    config_name="${config_names[$i]}"
    LORA_PATH="hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-${config_name}"
    NUM=$((i + 1))

    # Find the latest response file for this config
    INPUT_FILE=$(ls -t "${RESPONSES_DIR}/qwen-vl-8b-thinking-${config_name}"_*.json 2>/dev/null | head -1)

    if [ -z "$INPUT_FILE" ]; then
        echo "[$NUM/$TOTAL] SKIPPING $config_name: no response file found"
        FAILED+=("$config_name")
        continue
    fi

    # Extract timestamp from filename for output naming
    TIMESTAMP=$(basename "$INPUT_FILE" .json | sed "s/qwen-vl-8b-thinking-${config_name}_//")
    OUTPUT_DIR="${OUTPUT_BASE}/${config_name}"
    mkdir -p "$OUTPUT_DIR"

    CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_${TIMESTAMP}.json"
    CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_${TIMESTAMP}.json"
    EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_${TIMESTAMP}.json"

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] $config_name"
    echo "=========================================="
    echo "Input: $INPUT_FILE"
    echo "LoRA: $LORA_PATH"
    echo ""

    # 1. Confession evaluation
    echo "--- Confession ---"
    python confession_local.py \
        --model "$BASE_MODEL" \
        --input "$INPUT_FILE" \
        --output "$CONFESSION_OUTPUT" \
        --temperature $TEMPERATURE \
        --max-tokens $MAX_TOKENS_CONFESSION \
        --batch-size $BATCH_SIZE \
        --gpu-memory-utilization $GPU_MEMORY \
        --max-model-len $MAX_MODEL_LEN \
        --lora-adapter "$LORA_PATH" \
        --disable-compile

    if [ $? -ne 0 ]; then
        echo "[$NUM/$TOTAL] CONFESSION FAILED: $config_name"
        FAILED+=("$config_name")
        continue
    fi
    echo "Saved: $CONFESSION_OUTPUT"

    # 2. Classification evaluation
    echo "--- Classification ---"
    python classification_local.py \
        --model "$BASE_MODEL" \
        --input "$INPUT_FILE" \
        --output "$CLASSIFICATION_OUTPUT" \
        --temperature $TEMPERATURE \
        --max-tokens $MAX_TOKENS_CLASSIFICATION \
        --batch-size $BATCH_SIZE \
        --gpu-memory-utilization $GPU_MEMORY \
        --max-model-len $MAX_MODEL_LEN \
        --lora-adapter "$LORA_PATH" \
        --disable-compile

    if [ $? -ne 0 ]; then
        echo "[$NUM/$TOTAL] CLASSIFICATION FAILED: $config_name"
        FAILED+=("$config_name")
        continue
    fi
    echo "Saved: $CLASSIFICATION_OUTPUT"

    # 3. Evaluate confession (GPT-based classification)
    echo "--- Evaluate Confession ---"
    python evaluate_confession.py \
        --input "$CONFESSION_OUTPUT" \
        --output "$EVALUATED_CONFESSION_OUTPUT" \
        --evaluator-model "gpt-5.2" \
        --max-concurrent 10

    if [ $? -ne 0 ]; then
        echo "[$NUM/$TOTAL] EVALUATE CONFESSION FAILED: $config_name"
        FAILED+=("$config_name")
        continue
    fi
    echo "Saved: $EVALUATED_CONFESSION_OUTPUT"

    echo "[$NUM/$TOTAL] $config_name COMPLETE"
done

echo ""
echo "=========================================="
echo "ALL DONE at: $(date)"
echo "=========================================="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
else
    echo "All $TOTAL configs succeeded."
fi
