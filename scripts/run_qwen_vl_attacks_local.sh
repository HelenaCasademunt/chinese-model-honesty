#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

MODEL="Qwen/Qwen3-VL-8B-Thinking"
MODEL_NAME="qwen_qwen3_vl_8b_thinking"
QUESTIONS="data/dev_questions_explicit.json"
PREFILL_QUESTIONS="src/inference/prompts/assistant_prefill_dev_questions.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.9
MAX_MODEL_LEN=8192
BATCH_SIZE=100

mkdir -p results results/system_prompts results/pretrain_prompts

echo "=========================================="
echo "RUNNING LOCAL ATTACKS FOR ${MODEL}"
echo "=========================================="

# Baseline - COMPLETED
# echo ""
# echo "Running baseline evaluation..."
# python src/inference/local/baseline.py \
#     --model "$MODEL" \
#     --questions "$QUESTIONS" \
#     --output "results/${MODEL_NAME}_baseline.json" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens "$MAX_TOKENS" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

# Baseline no thinking - COMPLETED
# echo ""
# echo "Running baseline (no thinking) evaluation..."
# python src/inference/local/baseline_no_thinking.py \
#     --model "$MODEL" \
#     --questions "$QUESTIONS" \
#     --output "results/${MODEL_NAME}_baseline_no_thinking.json" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens "$MAX_TOKENS" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

# System prompt attack - COMPLETED
# echo ""
# echo "Running system prompt variations..."
# python src/inference/local/system_prompt.py \
#     --model "$MODEL" \
#     --questions "$QUESTIONS" \
#     --system-prompts "src/inference/prompts/system_prompts.json" \
#     --output-dir "results/system_prompts" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens "$MAX_TOKENS" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

# Assistant prefill attack - Standard prefills - COMPLETED
# echo ""
# echo "Running assistant prefill attack (standard prefills)..."
# python src/inference/local/assistant_prefill_attack.py \
#     --model "$MODEL" \
#     --questions "$PREFILL_QUESTIONS" \
#     --standard-prefills "src/inference/prompts/standard_prefills.json" \
#     --prefill-type answer \
#     --output "results/${MODEL_NAME}_assistant_prefill_standard.json" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens 2048 \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

# Assistant prefill attack - Finegrained prefills (embedded in questions) - COMPLETED
# echo ""
# echo "Running assistant prefill attack (finegrained prefills)..."
# python src/inference/local/assistant_prefill_attack.py \
#     --model "$MODEL" \
#     --questions "$PREFILL_QUESTIONS" \
#     --output "results/${MODEL_NAME}_assistant_prefill_finegrained.json" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens 2048 \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

# User prefill attack - COMPLETED
# echo ""
# echo "Running user prefill attack..."
# python src/inference/local/user_prefill_attack.py \
#     --model "$MODEL" \
#     --questions "$QUESTIONS" \
#     --user-prefills "src/inference/prompts/user_prefills.json" \
#     --output "results/${MODEL_NAME}_user_prefill.json" \
#     --temperature "$TEMPERATURE" \
#     --num-samples 5 \
#     --num-initial-samples 5 \
#     --max-tokens "$MAX_TOKENS" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size 4

# User prefill simple attack - COMPLETED
# echo ""
# echo "Running user prefill simple attack..."
# python src/inference/local/user_prefill_simple_attack.py \
#     --model "$MODEL" \
#     --questions "$QUESTIONS" \
#     --output "results/${MODEL_NAME}_user_prefill_simple.json" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens "$MAX_TOKENS" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

# Pretrain prompt attack
# echo ""
# echo "Running pretrain prompt attack..."
# python src/inference/local/pretrain_prompt_attack.py \
#     --model "$MODEL" \
#     --questions "$QUESTIONS" \
#     --prompts "src/inference/prompts/pretrain_prompts.json" \
#     --output-dir "results/pretrain_prompts" \
#     --temperature "$TEMPERATURE" \
#     --num-samples "$NUM_SAMPLES" \
#     --max-tokens "$MAX_TOKENS" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --batch-size "$BATCH_SIZE"

echo ""
echo "=========================================="
echo "All attacks complete."
echo "=========================================="

# Evaluations
echo ""
echo "=========================================="
echo "RUNNING EVALUATIONS"
echo "=========================================="

EVAL_CONFIGS_DIR="configs/qwen_vl_attacks"

for config in "$EVAL_CONFIGS_DIR"/*.yaml; do
    [ -f "$config" ] || continue
    config_name=$(basename "$config" .yaml)

    # Skip completed evals
    case "$config_name" in
        baseline|baseline_no_thinking|assistant_prefill*|user_prefill*)
            echo "Skipping $config_name (already completed)"
            continue
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/evaluation_pipeline.py "$config"
done

echo ""
echo "=========================================="
echo "All evaluations complete."
echo "=========================================="
