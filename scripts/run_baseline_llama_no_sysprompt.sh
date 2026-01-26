#!/bin/bash

source /root/.venv/bin/activate

MODEL="meta-llama/Llama-3.3-70B-Instruct"
OUTPUT="results/data/baseline_responses_llama70b_no_sysprompt.json"
EVALUATED_OUTPUT="results/data/evaluated_baseline_responses_llama70b_no_sysprompt.json"

# Run baseline with no system prompt
echo "Running baseline for Llama 3.3 70B (no system prompt)..."
python black_box/baseline.py \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --questions "black_box/eval/prompts/finegrained_eval_questions_filtered.json" \
  --num-samples 100 \
  --temperature 0.7 \
  --max-concurrent 5

echo "Baseline collection complete!"

# Run evaluation using OpenRouter
echo ""
echo "Evaluating responses..."
python black_box/eval/evaluate_openrouter.py \
  --attack-type "baseline" \
  --input "$OUTPUT" \
  --output "$EVALUATED_OUTPUT" \
  --facts "black_box/eval/prompts/finegrained_eval_facts_filtered.json" \
  --max-concurrent 10

echo ""
echo "========================================="
echo "All done!"
echo "========================================="
