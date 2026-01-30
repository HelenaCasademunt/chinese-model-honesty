#!/bin/bash
# Run all black_box attacks and LoRA evals using the new pipeline questions
# Results saved to results_new_pipeline/data/

QUESTIONS="black_box/eval/prompts/tiananmen_auditing_questions.json"
OUTDIR="results_new_pipeline/data"
NUM_SAMPLES=10
TEMPERATURE=0.7
MODEL="qwen/qwen3-32b"
MAX_TOKENS=3072
PROVIDER="deepinfra/fp8"
CONCURRENCY=50
BASE_MODEL="Qwen/Qwen3-32B"
BATCH_SIZE=50

mkdir -p "$OUTDIR"
mkdir -p "$OUTDIR/system_prompts"
mkdir -p "$OUTDIR/pretrain_prompts"

echo "========================================"
echo "Questions file: $QUESTIONS"
echo "Output dir: $OUTDIR"
echo "Model: $MODEL"
echo "Samples per question: $NUM_SAMPLES"
echo "========================================"

# 1. Baseline (must run first -- pretrain_attack depends on its output)
echo ""
echo "========================================"
echo "[1/7] Running baseline..."
echo "========================================"
python black_box/baseline.py \
    --questions "$QUESTIONS" \
    --output "$OUTDIR/baseline_responses.json" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --max-concurrent "$CONCURRENCY"

# 2. Assistant prefill attack (standard prefills, answer mode)
echo ""
echo "========================================"
echo "[2/7] Running assistant prefill attack..."
echo "========================================"
python black_box/assistant_prefill_attack.py \
    --questions "$QUESTIONS" \
    --standard-prefills black_box/prompts/standard_prefills.json \
    --prefill-type answer \
    --output "$OUTDIR/assistant_prefill_responses.json" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --provider "$PROVIDER" \
    --concurrency "$CONCURRENCY"

# 3. Pretrain attack (uses baseline responses)
echo ""
echo "========================================"
echo "[3/7] Running pretrain attack..."
echo "========================================"
python black_box/pretrain_attack.py \
    --questions "$QUESTIONS" \
    --baseline "$OUTDIR/baseline_responses.json" \
    --output "$OUTDIR/pretrain_attack_responses.json" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --provider "$PROVIDER" \
    --concurrency "$CONCURRENCY"

# 4. Pretrain prompt attack
echo ""
echo "========================================"
echo "[4/7] Running pretrain prompt attack..."
echo "========================================"
python black_box/pretrain_prompt_attack.py \
    --questions "$QUESTIONS" \
    --output-dir "$OUTDIR/pretrain_prompts" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --provider "$PROVIDER" \
    --concurrency "$CONCURRENCY"

# 5. User prefill attack
echo ""
echo "========================================"
echo "[5/7] Running user prefill attack..."
echo "========================================"
python black_box/user_prefill_attack.py \
    --questions "$QUESTIONS" \
    --user-prefills black_box/prompts/user_prefills.json \
    --output "$OUTDIR/user_prefill_attack_responses.json" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples 5 \
    --max-tokens "$MAX_TOKENS" \
    --provider "$PROVIDER" \
    --concurrency "$CONCURRENCY"

# 6. User prefill simple attack
echo ""
echo "========================================"
echo "[6/7] Running user prefill simple attack..."
echo "========================================"
python black_box/user_prefill_simple_attack.py \
    --questions "$QUESTIONS" \
    --output "$OUTDIR/user_prefill_simple_attack_responses.json" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples 5 \
    --max-tokens "$MAX_TOKENS" \
    --provider "$PROVIDER" \
    --concurrency "$CONCURRENCY"

# 7. System prompt attack
echo ""
echo "========================================"
echo "[7/7] Running system prompt attack..."
echo "========================================"
python black_box/system_prompt.py \
    --questions "$QUESTIONS" \
    --output-dir "$OUTDIR/system_prompts" \
    --temperature "$TEMPERATURE" \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --system-prompts black_box/prompts/system_prompts.json \
    --max-concurrent "$CONCURRENCY"

# 8. LoRA eval: mixed adapter
echo ""
echo "========================================"
echo "[8/9] Running eval with mixed LoRA adapter..."
echo "========================================"
python honesty_training/eval_local.py \
    --model "$BASE_MODEL" \
    --lora-adapter "hcasademunt/qwen3-32b-honesty-finetuned-mixed" \
    --questions "$QUESTIONS" \
    --output "$OUTDIR/lora_mixed_responses.json" \
    --temperature "$TEMPERATURE" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --batch-size "$BATCH_SIZE"

# 9. LoRA eval: goals adapter
echo ""
echo "========================================"
echo "[9/9] Running eval with goals LoRA adapter..."
echo "========================================"
python honesty_training/eval_local.py \
    --model "$BASE_MODEL" \
    --lora-adapter "hcasademunt/qwen3-32b-honesty-finetuned-goals" \
    --questions "$QUESTIONS" \
    --output "$OUTDIR/lora_goals_responses.json" \
    --temperature "$TEMPERATURE" \
    --num-samples "$NUM_SAMPLES" \
    --max-tokens "$MAX_TOKENS" \
    --batch-size "$BATCH_SIZE"

# 10. Convert responses to chinese_auditing format
echo ""
echo "========================================"
echo "[10] Converting responses to chinese_auditing format..."
echo "========================================"
python black_box/convert_responses.py \
    --input-dir "$OUTDIR" \
    --output-dir results_new_pipeline/converted \
    --model "$MODEL"

# 11. Run hypothesis auditor evaluation
echo ""
echo "========================================"
echo "[11] Running hypothesis auditor evaluation..."
echo "========================================"
bash black_box/run_evaluation.sh

echo ""
echo "========================================"
echo "All complete. Results in: $OUTDIR"
echo "========================================"
