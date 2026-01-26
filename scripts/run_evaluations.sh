#!/bin/bash

# Evaluation script for Chinese model honesty testing
# This script runs all evaluation types on previously generated responses


# Configuration
EVALUATOR_MODEL="gpt-5.2"
MAX_CONCURRENT=10

echo "========================================="
echo "EVALUATION PIPELINE"
echo "========================================="
echo "Evaluator: $EVALUATOR_MODEL"
echo ""


# =========================================
# SECTION 1: EVALUATE FINEGRAINED ASSISTANT PREFILL RESPONSES
# =========================================
echo ""
echo "========================================="
echo "SECTION 1: EVALUATE FINEGRAINED ASSISTANT PREFILL RESPONSES"
echo "========================================="

# 1.1: Evaluate with facts-based evaluation
echo ""
echo "1.1: Evaluating assistant finegrained prefill responses (facts-based)..."
python black_box/eval/evaluate_assistant_prefill_attacks_facts.py \
  --input "results/data/finegrained_assistant_prefill_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_finegrained_assistant_prefill_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Facts-based evaluation complete!"

# 1.2: Evaluate with standard evaluation
echo ""
echo "1.2: Evaluating assistant finegrained prefill responses (standard)..."
python black_box/eval/evaluate_assistant_prefill_attacks.py \
  --input "results/data/finegrained_assistant_prefill_responses.json" \
  --output "results/data/evaluated_finegrained_assistant_prefill_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"


# =========================================
# SECTION 3: EVALUATE USER PREFILL RESPONSES
# =========================================
echo ""
echo "========================================="
echo "SECTION 3: EVALUATE USER PREFILL RESPONSES"
echo "========================================="

# 3.1: Evaluate with facts-based evaluation
echo ""
echo "3.1: Evaluating user prefill attack responses (facts-based)..."
python black_box/eval/evaluate_user_prefill_attacks_facts.py \
  --input "results/data/user_prefill_attack_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_user_prefill_attack_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Facts-based evaluation complete!"

# 3.2: Evaluate with standard evaluation
echo ""
echo "3.2: Evaluating user prefill attack responses (standard)..."
python black_box/eval/evaluate_user_prefill_attacks.py \
  --input "results/data/user_prefill_attack_responses.json" \
  --output "results/data/evaluated_user_prefill_attack_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"


# =========================================
# SECTION 4: EVALUATE PRETRAIN ATTACK RESPONSES
# =========================================
echo ""
echo "========================================="
echo "SECTION 4: EVALUATE PRETRAIN ATTACK RESPONSES"
echo "========================================="

# 4.1: Evaluate with standard evaluation
echo ""
echo "4.1: Evaluating pretrain attack responses (standard)..."
python black_box/eval/evaluate_pretrain_attacks.py \
  --input "results/data/pretrain_attack_responses.json" \
  --output "results/data/evaluated_pretrain_attack_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"

# 4.2: Evaluate with facts-based evaluation
echo ""
echo "4.2: Evaluating pretrain attack responses (facts-based)..."
python black_box/eval/evaluate_pretrain_attacks_facts.py \
  --responses "results/data/pretrain_attack_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_pretrain_attack_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --concurrency $MAX_CONCURRENT

echo "Facts-based evaluation complete!"


# =========================================
# SECTION 2: EVALUATE STANDARD ASSISTANT PREFILL RESPONSES
# =========================================
echo ""
echo "========================================="
echo "SECTION 2: EVALUATE STANDARD ASSISTANT PREFILL RESPONSES"
echo "========================================="

# 2.1: Evaluate with standard evaluation
echo ""
echo "2.1: Evaluating assistant standard prefill responses (standard)..."
python black_box/eval/evaluate_assistant_prefill_attacks.py \
  --input "results/data/standard_assistant_prefill_responses.json" \
  --output "results/data/evaluated_standard_assistant_prefill_responses.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Standard evaluation complete!"

# 2.2: Evaluate with facts-based evaluation
echo ""
echo "2.2: Evaluating assistant standard prefill responses (facts-based)..."
python black_box/eval/evaluate_assistant_prefill_attacks_facts.py \
  --input "results/data/standard_assistant_prefill_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_standard_assistant_prefill_responses_facts.json" \
  --evaluator-model "$EVALUATOR_MODEL" \
  --max-concurrent $MAX_CONCURRENT

echo "Facts-based evaluation complete!"



# =========================================
# ALL DONE!
# =========================================
echo ""
echo "========================================="
echo "ALL EVALUATIONS COMPLETE!"
echo "========================================="
echo ""
echo "Evaluation results saved in results/data/:"
echo "  - evaluated_finegrained_assistant_prefill_responses.json"
echo "  - evaluated_finegrained_assistant_prefill_responses_facts.json"
echo "  - evaluated_standard_assistant_prefill_responses.json"
echo "  - evaluated_standard_assistant_prefill_responses_facts.json"
echo "  - evaluated_user_prefill_attack_responses.json"
echo "  - evaluated_user_prefill_attack_responses_facts.json"
echo "  - evaluated_pretrain_attack_responses.json"
echo "  - evaluated_pretrain_attack_responses_facts.json"
echo ""
