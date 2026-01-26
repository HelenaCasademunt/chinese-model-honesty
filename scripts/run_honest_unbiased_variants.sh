#!/bin/bash

source /root/.venv/bin/activate

VERSIONS=(
    "llama_prompt"
)

echo "Running system prompt attack for honest_unbiased variants..."
python black_box/system_prompt.py \
    --tags "${VERSIONS[@]}" \
    --output-dir results/data/system_prompts

echo ""
echo "Evaluating responses..."
for version in "${VERSIONS[@]}"; do
    echo "Evaluating: $version"
    python black_box/eval/evaluate_openrouter.py \
        --attack-type baseline \
        --input "results/data/system_prompts/responses_${version}.json" \
        --output "results/data/evaluated_baseline_responses_sys_${version}.json"
done

echo ""
echo "Done running and evaluating all honest_unbiased variants"
