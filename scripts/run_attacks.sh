#!/bin/bash

# Attack script for generating responses from Chinese models
# This script runs all attack types to collect model responses


# Configuration
MODEL="qwen/qwen3-32b"
PROVIDER="deepinfra/fp8"
TEMPERATURE=0.7

echo "========================================="
echo "ATTACK RESPONSE GENERATION PIPELINE"
echo "========================================="
echo "Model: $MODEL"
echo "Provider: $PROVIDER"
echo ""


# =========================================
# SECTION 1: ASSISTANT FINEGRAINED PREFILL ATTACKS (commented out)
# =========================================
# echo ""
# echo "========================================="
# echo "SECTION 1: ASSISTANT FINEGRAINED PREFILL ATTACKS"
# echo "========================================="

# 1.1: Generate responses with finegrained assistant prefills
# echo ""
# echo "1.1: Generating assistant finegrained prefill attack responses..."
# python black_box/assistant_prefill_attack.py \
#   --model "$MODEL" \
#   --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
#   --custom-prefills "black_box/prompts/finegrained_assistant_prefill.json" \
#   --output "results/data/finegrained_assistant_prefill_responses.json" \
#   --num-samples 100 \
#   --temperature $TEMPERATURE \
#   --max-tokens 1024 \
#   --provider "$PROVIDER" \
#   --max-concurrent-questions 5 \
#   --concurrency 20 \
#   --mode skip

# echo "Finegrained assistant prefill responses collected!"


# =========================================
# SECTION 2: USER PREFILL ATTACKS
# =========================================
echo ""
echo "========================================="
echo "SECTION 2: USER PREFILL ATTACKS"
echo "========================================="

# 2.1: Generate user prefill attack responses (10 initial answers, 10 per prefill)
echo ""
echo "2.1: Generating user prefill attack responses (10 initial answers, 10 per prefill)..."
python black_box/user_prefill_attack.py \
  --model "$MODEL" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --user-prefills "black_box/prompts/user_prefills.json" \
  --output "results/data/user_prefill_attack_responses.json" \
  --num-initial-samples 10 \
  --num-samples 10 \
  --temperature $TEMPERATURE \
  --max-tokens 1024 \
  --initial-max-tokens 3072 \
  --provider "$PROVIDER" \
  --max-concurrent-questions 3 \
  --concurrency 20 \
  --mode overwrite

echo "User prefill attack responses collected!"


# =========================================
# SECTION 3: PRETRAIN ATTACK
# =========================================
echo ""
echo "========================================="
echo "SECTION 3: PRETRAIN ATTACK"
echo "========================================="

# 3.1: Generate pretrain attack responses
echo ""
echo "3.1: Generating pretrain attack responses..."
python black_box/pretrain_attack.py \
  --model "$MODEL" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --baseline "results/data/baseline_responses.json" \
  --output "results/data/pretrain_attack_responses.json" \
  --num-samples 10 \
  --temperature $TEMPERATURE \
  --max-concurrent-questions 5 \
  --mode skip

echo "Pretrain attack responses collected!"

# =========================================
# SECTION 4: ASSISTANT STANDARD PREFILL ATTACKS
# =========================================
echo ""
echo "========================================="
echo "SECTION 4: ASSISTANT STANDARD PREFILL ATTACKS"
echo "========================================="

# 4.1: Generate responses with standard assistant prefills (100 completions per prefill)
echo ""
echo "4.1: Generating assistant standard prefill attack responses (100 completions per prefill)..."
python black_box/assistant_prefill_attack.py \
  --model "$MODEL" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --standard-prefills "black_box/prompts/standard_prefills.json" \
  --prefill-type "answer" \
  --output "results/data/standard_assistant_prefill_responses.json" \
  --num-samples 100 \
  --temperature $TEMPERATURE \
  --max-tokens 3072 \
  --provider "$PROVIDER" \
  --max-concurrent-questions 5 \
  --concurrency 20 \
  --mode overwrite

echo "Standard assistant prefill responses collected!"


# =========================================
# ALL DONE!
# =========================================
echo ""
echo "========================================="
echo "ALL ATTACK RESPONSES GENERATED!"
echo "========================================="
echo ""
echo "Response files saved in results/data/:"
echo "  - finegrained_assistant_prefill_responses.json"
echo "  - standard_assistant_prefill_responses.json"
echo "  - user_prefill_attack_responses.json"
echo "  - pretrain_attack_responses.json"
echo ""
echo "Now run ./run_evaluations.sh to evaluate the responses."
echo ""
