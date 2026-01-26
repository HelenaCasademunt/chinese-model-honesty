#!/bin/bash

# Run ask_if_true evaluation using Llama 3.3 70B as the evaluating model
# This uses the same question-answer pairs but has Llama assess truthfulness

source /root/.venv/bin/activate

echo "========================================="
echo "Running ask_if_true with Llama 3.3 70B"
echo "========================================="

# Step 1: Ask Llama if the responses are true and unbiased
echo ""
echo "Step 1: Asking Llama to evaluate truthfulness..."
python asking_the_model/ask_if_true.py \
  --input "results/data/evaluated_responses.json" \
  --output "results/data_asking_model/asking_model_truthfulness_llama.json" \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --temperature 0.7 \
  --max-tokens 1024 \
  --max-concurrent 50

echo ""
echo "Step 1 complete!"

# Step 2: Evaluate the asking results (classify TRUE/FALSE/OTHER)
echo ""
echo "Step 2: Evaluating asking results..."
python asking_the_model/evaluate_asking_results.py \
  --input "results/data_asking_model/asking_model_truthfulness_llama.json" \
  --output "results/data_asking_model/evaluated_asking_results_llama.json" \
  --evaluator-model "gpt-5.2" \
  --max-concurrent 50

echo ""
echo "Step 2 complete!"

echo ""
echo "========================================="
echo "All steps finished!"
echo "========================================="
echo "Results saved to:"
echo "  - results/data_asking_model/asking_model_truthfulness_llama.json"
echo "  - results/data_asking_model/evaluated_asking_results_llama.json"
