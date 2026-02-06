#!/bin/bash

source /root/.venv/bin/activate

cd /root/chinese-model-honesty

# Configuration
BASE_MODEL="Qwen/Qwen3-32B"
QUESTIONS="data/dev_questions.json"
TEMPERATURE=0.7
NUM_SAMPLES=10
BATCH_SIZE=4
MAX_TOKENS=2048
TENSOR_PARALLEL=1


# # Step 2: Finetune on control datasets
# echo ""
# echo "=========================================="
# echo "Step 2: Finetuning on control datasets"
# echo "=========================================="

# echo "Finetuning on OpenHermes..."
# python honesty_training/finetune_qwen3_32b.py \
#     --dataset-mode control \
#     --control-data honesty_training/data/openhermes_control.jsonl \
#     --num-samples 5000 \
#     --output-dir /workspace/qwen3-32b-control-openhermes \
#     --epochs 1 \
#     --hf-repo-id hcasademunt/qwen3-32b-control-openhermes

# echo ""
# echo "Finetuning on Alpaca..."
# python honesty_training/finetune_qwen3_32b.py \
#     --dataset-mode control \
#     --control-data honesty_training/data/alpaca_control.jsonl \
#     --num-samples 5000 \
#     --output-dir /workspace/qwen3-32b-control-alpaca \
#     --epochs 1 \
#     --hf-repo-id hcasademunt/qwen3-32b-control-alpaca

Step 3: Sample responses from finetuned models
echo ""
echo "=========================================="
echo "Step 3: Sampling responses from control models"
echo "=========================================="

CONTROL_ADAPTERS=(
    "/workspace/qwen3-32b-control-openhermes"
    "/workspace/qwen3-32b-control-alpaca"
)

for adapter_path in "${CONTROL_ADAPTERS[@]}"; do
    adapter_name=$(basename "$adapter_path")
    output_file="results/honesty/${adapter_name}.json"

    if [ ! -d "$adapter_path" ]; then
        echo "SKIPPING (not found): $adapter_path"
        continue
    fi

    echo ""
    echo "Sampling from: $adapter_name"
    echo "Output: $output_file"

    python honesty_eval/sample_assistant_responses_local.py \
        --model "$BASE_MODEL" \
        --lora-adapter "$adapter_path" \
        --questions "$QUESTIONS" \
        --output "$output_file" \
        --temperature $TEMPERATURE \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE \
        --max-tokens $MAX_TOKENS \
        --tensor-parallel-size $TENSOR_PARALLEL
done

# Step 4: Evaluate sampled responses
echo ""
echo "=========================================="
echo "Step 4: Evaluating responses"
echo "=========================================="

mkdir -p results/honesty/evaluated

for adapter_path in "${CONTROL_ADAPTERS[@]}"; do
    adapter_name=$(basename "$adapter_path")
    input_file="results/honesty/${adapter_name}.json"
    output_file="results/honesty/evaluated/${adapter_name}_evaluated.json"

    if [ ! -f "$input_file" ]; then
        echo "SKIPPING (input not found): $input_file"
        continue
    fi

    echo ""
    echo "Evaluating: $adapter_name"
    echo "Output: $output_file"

    python honesty_eval/evaluate.py \
        --attack-type baseline \
        --input "$input_file" \
        --output "$output_file" \
        --eval-mode both \
        --honesty-prompt-version 1
done

echo ""
echo "=========================================="
echo "All control dataset experiments complete!"
echo "Results in: results/honesty/"
echo "Evaluated results in: results/honesty/evaluated/"
echo "=========================================="
