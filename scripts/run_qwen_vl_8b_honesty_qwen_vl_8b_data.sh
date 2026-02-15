#!/bin/bash

source /root/.venv/bin/activate

########################################
# PHASE 0: DATA COLLECTION (OpenRouter API)
########################################

echo ""
echo "=========================================="
echo "PHASE 0: COLLECTING QWEN VL 8B RESPONSES"
echo "Started at: $(date)"
echo "=========================================="

cd /root/chinese-model-honesty

OR_MODEL="qwen/qwen3-vl-8b-instruct"
GOAL_INPUT="/root/chinese_auditing/src/honesty_finetuning/data/goal_data_honest_original_chat.jsonl"
FOLLOWUP_INPUT="/root/chinese_auditing/src/honesty_finetuning/data/followup_data_original_chat.jsonl"
DATA_DIR="honesty_training/data"

echo "=== Collecting goal + followup responses with Qwen VL 8B (in parallel) ==="

python honesty_training/collect_goal_responses.py \
    --goals "$GOAL_INPUT" \
    --source chat \
    --model "$OR_MODEL" \
    --output "$DATA_DIR/qwen_vl_8b_goal_responses.json" \
    --num-samples 1 \
    --max-concurrent 50 &
PID_GOAL=$!

python honesty_training/collect_followup_responses.py \
    --input "$FOLLOWUP_INPUT" \
    --source chat \
    --model "$OR_MODEL" \
    --output "$DATA_DIR/qwen_vl_8b_followup_responses.json" \
    --num-samples 1 \
    --max-concurrent 50 &
PID_FOLLOWUP=$!

echo "Goal PID: $PID_GOAL, Followup PID: $PID_FOLLOWUP"
wait $PID_GOAL
goal_exit=$?
echo "=== Goal collection finished (exit: $goal_exit) ==="
wait $PID_FOLLOWUP
followup_exit=$?
echo "=== Followup collection finished (exit: $followup_exit) ==="

if [ $goal_exit -ne 0 ] || [ $followup_exit -ne 0 ]; then
    echo "ERROR: Data collection failed. Aborting."
    exit 1
fi

echo ""
echo "=== Converting to chat format ==="
python honesty_training/format_deepseek_chat.py \
    --goal-input "$DATA_DIR/qwen_vl_8b_goal_responses.json" \
    --goal-output "$DATA_DIR/goal_data_qwen_vl_8b_chat.jsonl" \
    --followup-input "$DATA_DIR/qwen_vl_8b_followup_responses.json" \
    --followup-output "$DATA_DIR/followup_data_qwen_vl_8b_chat.jsonl"

convert_exit=$?
if [ $convert_exit -ne 0 ]; then
    echo "ERROR: Conversion failed. Aborting."
    exit 1
fi

echo "Goal data:     $DATA_DIR/goal_data_qwen_vl_8b_chat.jsonl"
echo "Followup data: $DATA_DIR/followup_data_qwen_vl_8b_chat.jsonl"

# Verify datasets are usable
echo ""
echo "=== Verifying datasets are usable ==="
for file in "$DATA_DIR/goal_data_qwen_vl_8b_chat.jsonl" "$DATA_DIR/followup_data_qwen_vl_8b_chat.jsonl"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
    line_count=$(wc -l < "$file")
    if [ "$line_count" -eq 0 ]; then
        echo "ERROR: File is empty: $file"
        exit 1
    fi
    echo "✓ $file ($line_count lines)"
    # Show first example
    echo "  First example:"
    head -1 "$file" | jq -r '.messages[0].content' | head -c 200
    echo "..."
done

echo "Data collection complete at: $(date)"

########################################
# TRAINING + SAMPLING + EVALUATION
########################################

cd /root/chinese_auditing

# Free up disk space from package caches
echo ""
echo "Cleaning package caches..."
uv cache clean 2>/dev/null
pip cache purge 2>/dev/null
echo "Disk after cleanup:"
df -h / | tail -1

BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"
QUESTIONS="data/dev_questions.json"
RESULTS_DIR="results/honesty"
LOG_DIR="logs/qwen_vl_8b_training"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODEL_CACHE="/root/model_cache/hub"

# Training configs
configs=(
    # Qwen VL 8B-generated data
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_goals_qwen_vl_8b.yaml
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_followup_qwen_vl_8b.yaml
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_mixed_qwen_vl_8b.yaml
    # Anthropic original data
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_goals_anthropic.yaml
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_followup_anthropic.yaml
    # Split personality data
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_followup_split_personality.yaml
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_split_personality_b_pass.yaml
    # Control datasets
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_control_alpaca.yaml
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_control_chinese_topics.yaml
    /root/chinese-model-honesty/honesty_training/input/qwen_vl_8b_control_openhermes.yaml
)

# Matching eval configs
eval_configs=(
    # Qwen VL 8B-generated data
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_goals_qwen_vl_8b.yaml
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_followup_qwen_vl_8b.yaml
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_mixed_qwen_vl_8b.yaml
    # Anthropic original data
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_goals_anthropic.yaml
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_followup_anthropic.yaml
    # Split personality data
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_followup_split_personality.yaml
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_split_personality_b_pass.yaml
    # Control datasets
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_control_alpaca.yaml
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_control_chinese_topics.yaml
    /root/chinese-model-honesty/scripts/configs/honesty_finetuning/eval_qwen_vl_8b_control_openhermes.yaml
)

TOTAL=${#configs[@]}
PASSED=0
FAILED=0
TRAIN_FAILED=()

########################################
# PHASE 1: ALL TRAININGS
########################################

echo ""
echo "=========================================="
echo "PHASE 1: ALL QWEN VL 8B TRAININGS"
echo "=========================================="

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    config_name=$(basename "$config" .yaml)
    log_file="$LOG_DIR/${config_name}_$(date +%Y%m%d_%H%M%S).log"
    NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Training: $config_name"
    echo "Log: $log_file"
    echo "Started at: $(date)"
    echo "=========================================="

    python src/honesty_finetuning/finetune_honesty.py "$config" 2>&1 | tee "$log_file"
    train_exit=${PIPESTATUS[0]}

    if [ $train_exit -ne 0 ]; then
        echo "[$NUM/$TOTAL] TRAINING FAILED: $config_name (exit code: $train_exit)"
        TRAIN_FAILED+=("$i")
        echo ""
        continue
    fi

    echo "[$NUM/$TOTAL] Training succeeded: $config_name"
done

########################################
# PHASE 2: ALL SAMPLINGS (vLLM, full model)
########################################

echo ""
echo "=========================================="
echo "PHASE 2: ALL QWEN VL 8B SAMPLINGS"
echo "=========================================="

for i in "${!configs[@]}"; do
    # Skip configs that failed training
    skip=false
    for fi in "${TRAIN_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then
            skip=true
            break
        fi
    done
    if $skip; then
        config_name=$(basename "${configs[$i]}" .yaml)
        echo "Skipping sampling for $config_name (training failed)"
        FAILED=$((FAILED + 1))
        continue
    fi

    config="${configs[$i]}"
    config_name=$(basename "$config" .yaml)
    NUM=$((i + 1))

    OUTPUT_DIR=$(grep "^output_dir:" "$config" | awk '{print $2}')
    LORA_NAME="${config_name#qwen_vl_8b_}"

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Sampling responses: $config_name"
    echo "=========================================="

    python src/honesty_finetuning/sample_assistant_responses_local.py \
        --model "$BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$RESULTS_DIR/qwen-vl-8b-${LORA_NAME}.json" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 200 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$OUTPUT_DIR"

    sample_exit=$?

    if [ $sample_exit -ne 0 ]; then
        echo "[$NUM/$TOTAL] SAMPLING FAILED: $config_name (exit code: $sample_exit)"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "[$NUM/$TOTAL] Sampling succeeded: $RESULTS_DIR/qwen-vl-8b-${LORA_NAME}.json"
done

########################################
# PHASE 3: ALL EVALUATIONS (API only, no GPU)
########################################

echo ""
echo "=========================================="
echo "PHASE 3: ALL EVALUATIONS"
echo "=========================================="

for i in "${!configs[@]}"; do
    # Skip configs that failed training
    skip=false
    for fi in "${TRAIN_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then
            skip=true
            break
        fi
    done
    if $skip; then
        continue
    fi

    config="${configs[$i]}"
    eval_config="${eval_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    LORA_NAME="${config_name#qwen_vl_8b_}"
    NUM=$((i + 1))

    # Check that response file exists
    response_file="$RESULTS_DIR/qwen-vl-8b-${LORA_NAME}.json"
    if [ ! -f "$response_file" ]; then
        echo "[$NUM/$TOTAL] SKIPPING EVAL: $config_name (response file missing: $response_file)"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/evaluation_pipeline.py "$eval_config"
    eval_exit=$?

    if [ $eval_exit -eq 0 ]; then
        echo "[$NUM/$TOTAL] PASSED: $config_name"
        PASSED=$((PASSED + 1))
    else
        echo "[$NUM/$TOTAL] EVALUATION FAILED: $config_name (exit code: $eval_exit)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
echo "Results: $PASSED/$TOTAL passed, $FAILED failed"
