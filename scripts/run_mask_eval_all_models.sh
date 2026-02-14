#!/bin/bash

source /root/.venv/bin/activate

cd /root/chinese-model-honesty

CURRENT_DATE="20260205"

# Base model
BASE_MODEL="Qwen/Qwen3-32B"

# Common settings
TENSOR_PARALLEL=1
BATCH_SIZE=8
GPU_MEM=0.95
MAX_MODEL_LEN=8192

# LoRA adapter directories from honesty training configs
LORA_ADAPTERS=(
    "/workspace/qwen3-32b-lora-finetuned-honesty-split-personality"
    "/workspace/qwen3-32b-lora-finetuned-followup-split-personality-data"
    "/workspace/qwen3-32b-lora-finetuned-followup-anthropic"
    "/workspace/qwen3-32b-lora-finetuned-followup-qwen-data"
    "/workspace/qwen3-32b-lora-finetuned-goals-anthropic"
    "/workspace/qwen3-32b-lora-finetuned-goals-qwen"
    "/workspace/qwen3-32b-lora-finetuned-mixed-anthropic"
    "/workspace/qwen3-32b-lora-finetuned-mixed-qwen"
    "/workspace/qwen3-32b-lora-finetuned-mixed-split-personality"
    "/workspace/qwen3-32b-split-personality-${CURRENT_DATE}"
    "/workspace/qwen3-32b-split-personality-a-prompt-${CURRENT_DATE}"
)

# Create results directory
mkdir -p mask_control/results

# First, evaluate the base model without any LoRA adapter
echo "=========================================="
echo "Evaluating base model: $BASE_MODEL"
echo "=========================================="
python mask_control/eval_mask.py \
    --model "$BASE_MODEL" \
    --output "mask_control/results/mask_eval_baseline.json" \
    --config known_facts provided_facts \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --batch-size $BATCH_SIZE \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_MODEL_LEN

# Evaluate each LoRA adapter
for adapter_path in "${LORA_ADAPTERS[@]}"; do
    adapter_name=$(basename "$adapter_path")
    output_file="mask_control/results/mask_eval_${adapter_name}.json"

    # Check if adapter directory exists
    if [ ! -d "$adapter_path" ]; then
        echo "=========================================="
        echo "SKIPPING (not found): $adapter_path"
        echo "=========================================="
        continue
    fi

    echo "=========================================="
    echo "Evaluating: $adapter_name"
    echo "Adapter path: $adapter_path"
    echo "Output: $output_file"
    echo "=========================================="

    python mask_control/eval_mask.py \
        --model "$BASE_MODEL" \
        --lora-adapter "$adapter_path" \
        --output "$output_file" \
        --config known_facts provided_facts \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --batch-size $BATCH_SIZE \
        --gpu-memory-utilization $GPU_MEM \
        --max-model-len $MAX_MODEL_LEN
done

echo "=========================================="
echo "All MASK evaluations complete!"
echo "Results saved to mask_control/results/"
echo "=========================================="

# Now run check_response_match.py for all result files
echo ""
echo "=========================================="
echo "Running check_response_match for all results"
echo "=========================================="

for result_file in mask_control/results/mask_eval_*_known_facts.json; do
    if [ -f "$result_file" ]; then
        echo ""
        echo "--- Checking: $result_file ---"
        python mask_control/check_response_match.py --input "$result_file"
    fi
done

echo ""
echo "=========================================="
echo "All checks complete!"
echo "=========================================="
