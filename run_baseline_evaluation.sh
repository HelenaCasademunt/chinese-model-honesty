#!/bin/bash

# Run baseline evaluation for Qwen 3 32B
echo "Running baseline evaluation for Qwen 3 32B..."
python black_box/baseline.py \
  --model "qwen/qwen3-32b" \
  --output "results/data/baseline_responses.json" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --num-samples 100 \
  --temperature 0.7 \
  --max-concurrent 50

echo "Qwen evaluation complete!"

# Run baseline evaluation for Llama 3.3 70B
echo "Running baseline evaluation for Llama 3.3 70B..."
python black_box/baseline.py \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --output "results/data/baseline_responses_llama70b.json" \
  --questions "black_box/eval/prompts/finegrained_eval_questions.json" \
  --num-samples 100 \
  --temperature 0.7 \
  --max-concurrent 50

echo "Llama evaluation complete!"
echo "All evaluations finished!"
