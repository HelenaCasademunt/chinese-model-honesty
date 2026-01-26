#!/bin/bash

# Run baseline evaluation for Qwen 3 32B
echo "Running baseline evaluation for Qwen 3 32B..."
python black_box/baseline.py \
  --model "qwen/qwen3-32b" \
  --output "results/data/baseline_responses.json" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --num-samples 100 \
  --temperature 0.7 \
  --max-concurrent 5 \
  --mode overwrite

echo "Qwen evaluation complete!"

# Run baseline evaluation for Llama 3.3 70B
echo "Running baseline evaluation for Llama 3.3 70B..."
python black_box/baseline.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --output "results/data/baseline_responses_llama70b.json" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --num-samples 100 \
  --temperature 0.7 \
  --max-concurrent 5

echo "Llama evaluation complete!"

# Evaluate Qwen responses using truthfulness evaluation
echo ""
echo "Evaluating Qwen responses (truthfulness)..."
python black_box/eval/evaluate_truthfulness.py \
  --input "results/data/baseline_responses.json" \
  --output "results/data/evaluated_responses.json" \
  --evaluator-model "gpt-5.2" \
  --max-concurrent 10

echo "Qwen truthfulness evaluation complete!"

# Evaluate Llama responses using truthfulness evaluation
echo ""
echo "Evaluating Llama responses (truthfulness)..."
python black_box/eval/evaluate_truthfulness.py \
  --input "results/data/baseline_responses_llama70b.json" \
  --output "results/data/evaluated_responses_llama70b.json" \
  --evaluator-model "gpt-5.2" \
  --max-concurrent 10

echo "Llama truthfulness evaluation complete!"

# Evaluate Qwen responses using facts-based evaluation
echo ""
echo "Evaluating Qwen responses (facts-based)..."
python black_box/eval/evaluate_truthfulness_facts.py \
  --input "results/data/baseline_responses.json" \
  --output "results/data/evaluated_responses_facts.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --evaluator-model "gpt-5-mini" \
  --max-concurrent 10

echo "Qwen facts-based evaluation complete!"

# Evaluate Llama responses using facts-based evaluation
echo ""
echo "Evaluating Llama responses (facts-based)..."
python black_box/eval/evaluate_truthfulness_facts.py \
  --input "results/data/baseline_responses_llama70b.json" \
  --output "results/data/evaluated_responses_facts_llama70b.json" \
  --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
  --evaluator-model "gpt-5-mini" \
  --max-concurrent 10

echo "Llama facts-based evaluation complete!"

echo ""
echo "========================================="
echo "All evaluations finished!"
echo "========================================="
