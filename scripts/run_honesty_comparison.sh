#!/bin/bash

source /root/.venv/bin/activate

cd /root/chinese-model-honesty

INPUT_FILE="results/llama-70b-baseline_responses.json"

echo "Running honesty evaluation with prompt version 1 (honesty only)..."
python honesty_eval/evaluate.py \
  --attack-type simple \
  --input "$INPUT_FILE" \
  --output "results/eval_llama_honesty_v1.json" \
  --eval-mode honesty \
  --honesty-prompt-version 1 \
  --max-concurrent 5

echo "Running honesty evaluation with prompt version 2 (with fact extraction)..."
python honesty_eval/evaluate.py \
  --attack-type simple \
  --input "$INPUT_FILE" \
  --output "results/eval_llama_honesty_v2_with_facts.json" \
  --eval-mode both \
  --honesty-prompt-version 2 \
  --max-concurrent 5

echo "All evaluations complete!"
