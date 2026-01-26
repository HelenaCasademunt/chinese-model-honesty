#!/bin/bash

source /root/.venv/bin/activate

# Run baseline with 100 answers per question
echo "Running baseline (100 samples)..."
python black_box/baseline.py \
    --model "qwen/qwen3-32b" \
    --output "results/data/baseline_responses_100.json" \
    --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
    --num-samples 100 \
    --temperature 0.7 \
    --max-concurrent 5

# Run standard evaluation
echo ""
echo "Running standard evaluation..."
python black_box/eval/evaluate_truthfulness.py \
    --input "results/data/baseline_responses_100.json" \
    --output "results/data/evaluated_responses_100.json" \
    --evaluator-model "gpt-5.2" \
    --max-concurrent 10

# Run facts only evaluation
echo ""
echo "Running facts only evaluation..."
python black_box/eval/evaluate_facts_only.py \
    --input "results/data/evaluated_responses_100.json" \
    --output "results/data/evaluated_responses_facts_only_100.json" \
    --facts "black_box/eval/prompts/finegrained_eval_facts.json" \
    --max-concurrent 60

echo ""
echo "Done!"
