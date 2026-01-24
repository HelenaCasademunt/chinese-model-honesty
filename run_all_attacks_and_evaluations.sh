#!/bin/bash

# Comprehensive evaluation and attack script for Chinese model honesty testing
# This script runs all attack types and their evaluations

set -e  # Exit on error

# Configuration
MODEL="qwen/qwen3-32b"
PROVIDER="deepinfra/fp8"
EVALUATOR_MODEL="gpt-5.2"
TEMPERATURE=0.7
MAX_CONCURRENT=10

echo "========================================="
echo "COMPREHENSIVE ATTACK & EVALUATION PIPELINE"
echo "========================================="
echo "Model: $MODEL"
echo "Provider: $PROVIDER"
echo "Evaluator: $EVALUATOR_MODEL"
echo ""


# Evaluate Qwen responses using facts-based evaluation
echo ""
echo "Evaluating Qwen responses (facts-based)..."
python black_box/eval/evaluate_truthfulness_facts.py \
  --input "results/data/baseline_responses.json" \
  --output "results/data/evaluated_responses_facts.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --evaluator-model "gpt-5.2" \
  --max-concurrent 10

echo "Qwen facts-based evaluation complete!"

# Evaluate Llama responses using facts-based evaluation
echo ""
echo "Evaluating Llama responses (facts-based)..."
python black_box/eval/evaluate_truthfulness_facts.py \
  --input "results/data/baseline_responses_llama70b.json" \
  --output "results/data/evaluated_responses_facts_llama70b.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --evaluator-model "gpt-5.2" \
  --max-concurrent 10

echo "Llama facts-based evaluation complete!"


# =========================================
# SECTION 1: ASSISTANT FINEGRAINED PREFILL ATTACKS
# =========================================
echo ""
echo "========================================="
echo "SECTION 1: ASSISTANT FINEGRAINED PREFILL ATTACKS"
echo "========================================="

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

# 1.2: Evaluate with facts-based evaluation
echo ""
echo "1.2: Evaluating assistant finegrained prefill responses (facts-based)..."
python black_box/eval/evaluate_assistant_prefill_attacks_facts.py \
  --responses "results/data/finegrained_assistant_prefill_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_finegrained_assistant_prefill_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --concurrency $MAX_CONCURRENT

echo "Facts-based evaluation complete!"

# 1.3: Evaluate with standard evaluation
echo ""
echo "1.3: Evaluating assistant finegrained prefill responses (standard)..."
python black_box/eval/evaluate_assistant_prefill_attacks.py \
  --input "results/data/finegrained_assistant_prefill_responses.json" \
  --output "results/data/evaluated_finegrained_assistant_prefill_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"

# =========================================
# SECTION 2: ASSISTANT STANDARD PREFILL ATTACKS
# =========================================
echo ""
echo "========================================="
echo "SECTION 2: ASSISTANT STANDARD PREFILL ATTACKS"
echo "========================================="

# 2.1: Generate responses with standard assistant prefills (100 completions per prefill)
echo ""
echo "2.1: Generating assistant standard prefill attack responses (100 completions per prefill)..."
python black_box/assistant_prefill_attack.py \
  --model "$MODEL" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --standard-prefills "black_box/prompts/standard_prefills.json" \
  --prefill-type "answer" \
  --output "results/data/standard_assistant_prefill_responses.json" \
  --num-samples 100 \
  --temperature $TEMPERATURE \
  --max-tokens 1024 \
  --provider "$PROVIDER" \
  --max-concurrent-questions 5 \
  --concurrency 20 \
  --mode overwrite

echo "Standard assistant prefill responses collected!"

# 2.2: Evaluate with standard evaluation
echo ""
echo "2.2: Evaluating assistant standard prefill responses (standard)..."
python black_box/eval/evaluate_assistant_prefill_attacks.py \
  --input "results/data/standard_assistant_prefill_responses.json" \
  --output "results/data/evaluated_standard_assistant_prefill_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"

# 2.3: Evaluate with facts-based evaluation
echo ""
echo "2.3: Evaluating assistant standard prefill responses (facts-based)..."
python black_box/eval/evaluate_assistant_prefill_attacks_facts.py \
  --responses "results/data/standard_assistant_prefill_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_standard_assistant_prefill_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --concurrency $MAX_CONCURRENT

echo "Facts-based evaluation complete!"

# =========================================
# SECTION 3: USER PREFILL ATTACKS
# =========================================
echo ""
echo "========================================="
echo "SECTION 3: USER PREFILL ATTACKS"
echo "========================================="

# 3.1: Generate user prefill attack responses (10 initial answers, 10 per prefill)
echo ""
echo "3.1: Generating user prefill attack responses (10 initial answers, 10 per prefill)..."
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

# 3.2: Evaluate with facts-based evaluation
echo ""
echo "3.2: Evaluating user prefill attack responses (facts-based)..."
python black_box/eval/evaluate_user_prefill_attacks_facts.py \
  --responses "results/data/user_prefill_attack_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_user_prefill_attack_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --concurrency $MAX_CONCURRENT

echo "Facts-based evaluation complete!"

# 3.3: Evaluate with standard evaluation
echo ""
echo "3.3: Evaluating user prefill attack responses (standard)..."
python black_box/eval/evaluate_user_prefill_attacks.py \
  --input "results/data/user_prefill_attack_responses.json" \
  --output "results/data/evaluated_user_prefill_attack_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"

# =========================================
# SECTION 4: PRETRAIN ATTACK
# =========================================
echo ""
echo "========================================="
echo "SECTION 4: PRETRAIN ATTACK"
echo "========================================="

# 4.1: Generate pretrain attack responses
echo ""
echo "4.1: Generating pretrain attack responses..."
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

# 4.2: Evaluate with standard evaluation
echo ""
echo "4.2: Evaluating pretrain attack responses (standard)..."
python black_box/eval/evaluate_pretrain_attacks.py \
  --input "results/data/pretrain_attack_responses.json" \
  --output "results/data/evaluated_pretrain_attack_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"

# 4.3: Evaluate with facts-based evaluation
echo ""
echo "4.3: Evaluating pretrain attack responses (facts-based)..."
python black_box/eval/evaluate_pretrain_attacks_facts.py \
  --responses "results/data/pretrain_attack_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_pretrain_attack_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --concurrency $MAX_CONCURRENT

echo "Facts-based evaluation complete!"

# =========================================
# ALL DONE!
# =========================================
echo ""
echo "========================================="
echo "ALL ATTACKS AND EVALUATIONS COMPLETE!"
echo "========================================="
echo ""
echo "Results saved in results/data/ directory:"
echo "  - finegrained_assistant_prefill_responses.json"
echo "  - evaluated_finegrained_assistant_prefill_responses.json"
echo "  - evaluated_finegrained_assistant_prefill_responses_facts.json"
echo "  - assistant_prefill_responses.json"
echo "  - evaluated_assistant_prefill_responses.json"
echo "  - evaluated_assistant_prefill_responses_facts.json"
echo "  - user_prefill_attack_responses.json"
echo "  - evaluated_user_prefill_attack_responses.json"
echo "  - evaluated_user_prefill_attack_responses_facts.json"
echo "  - pretrain_attack_responses.json"
echo "  - evaluated_pretrain_attack_responses.json"
echo "  - evaluated_pretrain_attack_responses_facts.json"
echo ""
