#!/bin/bash

# Run pretrain attack evaluation for Qwen 3 32B
echo "Running pretrain attack evaluation for Qwen 3 32B..."
python black_box/pretrain_attack.py \
  --model "qwen/qwen3-32b" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --baseline "results/data/baseline_responses.json" \
  --output "results/data/pretrain_attack_responses.json" \
  --num-samples 10 \
  --temperature 0.7 \
  --max-concurrent-questions 5 \
  --mode skip

echo "Pretrain attack data collection complete!"

# Evaluate pretrain attack responses using simple evaluation
echo ""
echo "Evaluating pretrain attack responses (simple)..."
python black_box/eval/evaluate_pretrain_attacks.py \
  --input "results/data/pretrain_attack_responses.json" \
  --output "results/data/evaluated_pretrain_attack_responses.json" \
  --evaluator-model "gpt-4o-mini" \
  --max-concurrent 10

echo "Simple evaluation complete!"

# Evaluate pretrain attack responses using facts-based evaluation
echo ""
echo "Evaluating pretrain attack responses (facts-based)..."
python black_box/eval/evaluate_pretrain_attacks_facts.py \
  --responses "results/data/pretrain_attack_responses.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --output "results/data/evaluated_pretrain_attack_responses_facts.json" \
  --evaluator-model "gpt-4o-mini" \
  --concurrency 10

echo "Facts-based evaluation complete!"

echo ""
echo "========================================="
echo "Pretrain attack evaluation finished!"
echo "========================================="
