#!/bin/bash

source /root/.venv/bin/activate

VERSIONS=(
    "honest_unbiased_v1"
    "honest_unbiased_v2"
    "honest_unbiased_v3"
    "honest_unbiased_v4"
    "honest_unbiased_v5"
)

for version in "${VERSIONS[@]}"; do
    echo "Evaluating system prompt: $version"
    python black_box/eval/evaluate_openrouter.py \
        --attack-type baseline \
        --input "results/data/system_prompts/responses_${version}.json" \
        --output "results/data/evaluated_baseline_responses_sys_${version}.json"
done

echo "Done evaluating all system prompt versions"
