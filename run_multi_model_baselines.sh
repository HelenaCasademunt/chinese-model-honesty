#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

QUESTIONS="data/dev_questions.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000

mkdir -p results

# ── Model definitions ────────────────────────────────────────────────
# Format: OPENROUTER_ID|SHORT_NAME|THINKING (yes/no)
MODELS=(
    "deepseek/deepseek-r1-distill-llama-8b|deepseek_r1_distill_llama_8b|yes"
    "deepseek/deepseek-r1-distill-qwen-32b|deepseek_r1_distill_qwen_32b|yes"
    "thudm/glm-4-32b|glm_4_32b_0414|no"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_NAME THINKING <<< "$entry"

    echo ""
    echo "############################################################"
    echo "# Model: $MODEL"
    echo "############################################################"

    # ── Baseline (with thinking for thinking models, standard for others) ──
    echo ""
    echo "=========================================="
    echo "Running baseline for $MODEL"
    echo "=========================================="
    python src/inference/baseline.py \
        --questions "$QUESTIONS" \
        --model "$MODEL" \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$MAX_TOKENS" \
        --output "results/${MODEL_NAME}_baseline.json"

    # ── No-thinking variant (only for thinking models) ──
    if [ "$THINKING" = "yes" ]; then
        echo ""
        echo "=========================================="
        echo "Running baseline no-thinking for $MODEL"
        echo "=========================================="
        python src/inference/baseline_no_thinking.py \
            --questions "$QUESTIONS" \
            --model "$MODEL" \
            --num-samples "$NUM_SAMPLES" \
            --temperature "$TEMPERATURE" \
            --max-tokens "$MAX_TOKENS" \
            --output "results/${MODEL_NAME}_no_thinking.json"
    else
        echo ""
        echo "(Skipping no-thinking — $MODEL is not a thinking model)"
    fi
done

# ── Evaluations ──────────────────────────────────────────────────────
echo ""
echo "############################################################"
echo "# RUNNING EVALUATIONS"
echo "############################################################"

EVAL_CONFIGS_DIR="configs/multi_model_baselines"

for config in "$EVAL_CONFIGS_DIR"/*.yaml; do
    [ -f "$config" ] || continue
    config_name=$(basename "$config" .yaml)

    echo ""
    echo "=========================================="
    echo "Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/evaluation_pipeline.py "$config"
done

echo ""
echo "All done."
