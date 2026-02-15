#!/usr/bin/env python3
"""
Generate evaluation config files for attack response files.

This script creates eval configs for:
- Assistant prefill attacks (2 prefills for "answer" mode)
- User prefill attacks (4 prefills)
- User prefill simple attack (1 file, no prefill variants)
- System prompt attacks (11 variants)
- Pretrain prompt attacks (3 variants)
"""

import os
import json
from pathlib import Path

# Base configuration template
EVAL_CONFIG_TEMPLATE = """# Configuration for Response Evaluation Pipeline
# Usage: python src/evaluation/evaluation_pipeline.py configs/response_evaluation.yaml
#
# This pipeline evaluates model responses for:
# 1. Refusal detection (yes/no)
# 2. Honesty scoring (1-100)
# 3. Fact verification (yes/no/lie for each ground truth fact)

# Input files
responses_file: "{responses_file}"
facts_file: "data/dev_facts_explicit.json"
# Output directory
output_dir: "{output_dir}"
# Limit number of responses (null = use all)
max_responses: null
# Minimum fact count threshold: filter out ground truth facts with count below this value (null = no filtering)
min_fact_count: 3

# Evaluation models (use full model names, "openai/" prefix is stripped for batch API)
# Can specify per-step models, or use a single default model
models:
  default: "google/gemini-3-flash-preview"
  refusal: "google/gemini-3-flash-preview"
  honesty: "google/gemini-3-flash-preview"
  fact_verification: "google/gemini-3-flash-preview"
  hypothesis_extraction: "openai/gpt-5.2"


# API selection per step: "batch" (OpenAI Batch API) or "openrouter" (real-time)
api:
  default: "openrouter"

# Common settings
temperature: 1.0

# OpenRouter reasoning/extended thinking (only used for steps with api: "openrouter")
reasoning:
  enabled: true
  effort: "medium"

# OpenRouter settings (only used for steps with api: "openrouter")
max_concurrent: 50
max_retries: 10
retry_delay: 1.0

# Skip individual steps (default: false)
skip_refusal: false
skip_honesty: false
skip_fact_verification: false
skip_hypothesis_extraction: true

# Task-specific max_tokens (minimum 16 for OpenAI models)
refusal:
  max_tokens: 10000

honesty:
  max_tokens: 10000

fact_verification:
  max_tokens: 10000

hypothesis_extraction:
  max_tokens: 10000

# Batch API settings (only used for steps with api: "batch")
batch:
  poll_interval: 30
  timeout: 86400
"""

# Model name and base paths
MODEL_NAME = "qwen_qwen3_vl_8b_thinking"
RESULTS_BASE = f"/root/chinese_auditing/results/{MODEL_NAME}"
CONFIG_OUTPUT_DIR = "/root/chinese-model-honesty/scripts/configs/honesty_finetuning"

# Attack configurations
ATTACK_CONFIGS = {
    # Assistant prefill (2 answer prefills)
    "assistant_prefill": {
        "num_files": 2,
        "file_pattern": f"{MODEL_NAME}_assistant_prefill_prefill_{{idx}}.json",
        "output_pattern": f"qwen3_vl_8b_thinking/assistant_prefill_{{idx}}",
        "config_name_pattern": "eval_qwen3_vl_8b_thinking_assistant_prefill_{idx}.yaml",
    },
    # User prefill (4 user prefills)
    "user_prefill": {
        "num_files": 4,
        "file_pattern": f"{MODEL_NAME}_user_prefill_prefill_{{idx}}.json",
        "output_pattern": f"qwen3_vl_8b_thinking/user_prefill_{{idx}}",
        "config_name_pattern": "eval_qwen3_vl_8b_thinking_user_prefill_{idx}.yaml",
    },
    # User prefill simple (1 file, no variants)
    "user_prefill_simple": {
        "num_files": 1,
        "file_pattern": f"{MODEL_NAME}_user_prefill_simple.json",
        "output_pattern": f"qwen3_vl_8b_thinking/user_prefill_simple",
        "config_name_pattern": "eval_qwen3_vl_8b_thinking_user_prefill_simple.yaml",
        "single_file": True,
    },
}

# System prompts (11 variants)
SYSTEM_PROMPTS = [
    "none", "helpful_assistant", "honest_transparent", "honest_unbiased",
    "honest_unbiased_you", "honest_unbiased_v1", "honest_unbiased_v2",
    "honest_unbiased_v3", "honest_unbiased_v4", "honest_unbiased_v5",
    "chinese_aligned", "llama_prompt"
]

# Pretrain prompts (3 variants)
PRETRAIN_PROMPTS = [
    "censored_chinese_ai", "author_roleplay", "claude_transcript"
]


def create_eval_config(responses_file: str, output_dir: str, config_path: str):
    """Create an evaluation config file."""
    config_content = EVAL_CONFIG_TEMPLATE.format(
        responses_file=responses_file,
        output_dir=output_dir,
    )

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"Created: {config_path}")


def main():
    created_count = 0

    print("Generating evaluation config files...\n")

    # Generate configs for assistant/user prefill attacks
    for attack_name, config in ATTACK_CONFIGS.items():
        print(f"=== {attack_name} ===")

        if config.get("single_file"):
            # Single file, no variants
            responses_file = os.path.join(RESULTS_BASE, config["file_pattern"])
            output_dir = f"output/evaluation/{config['output_pattern']}"
            config_path = os.path.join(CONFIG_OUTPUT_DIR, config["config_name_pattern"])

            create_eval_config(responses_file, output_dir, config_path)
            created_count += 1
        else:
            # Multiple files with prefill variants
            for idx in range(config["num_files"]):
                responses_file = os.path.join(RESULTS_BASE, config["file_pattern"].format(idx=idx))
                output_dir = f"output/evaluation/{config['output_pattern'].format(idx=idx)}"
                config_path = os.path.join(CONFIG_OUTPUT_DIR, config["config_name_pattern"].format(idx=idx))

                create_eval_config(responses_file, output_dir, config_path)
                created_count += 1

        print()

    # Generate configs for system prompt attacks
    print("=== system_prompts ===")
    model_name_safe = MODEL_NAME.replace("/", "_").replace("-", "_")
    for prompt_tag in SYSTEM_PROMPTS:
        responses_file = os.path.join(
            RESULTS_BASE, "system_prompts", f"{model_name_safe}_system_{prompt_tag}.json"
        )
        output_dir = f"output/evaluation/qwen3_vl_8b_thinking/system_{prompt_tag}"
        config_path = os.path.join(
            CONFIG_OUTPUT_DIR, f"eval_qwen3_vl_8b_thinking_system_{prompt_tag}.yaml"
        )

        create_eval_config(responses_file, output_dir, config_path)
        created_count += 1
    print()

    # Generate configs for pretrain prompt attacks
    print("=== pretrain_prompts ===")
    for prompt_tag in PRETRAIN_PROMPTS:
        responses_file = os.path.join(
            RESULTS_BASE, "pretrain_prompts", f"{model_name_safe}_pretrain_{prompt_tag}.json"
        )
        output_dir = f"output/evaluation/qwen3_vl_8b_thinking/pretrain_{prompt_tag}"
        config_path = os.path.join(
            CONFIG_OUTPUT_DIR, f"eval_qwen3_vl_8b_thinking_pretrain_{prompt_tag}.yaml"
        )

        create_eval_config(responses_file, output_dir, config_path)
        created_count += 1
    print()

    print(f"{'='*60}")
    print(f"Total configs created: {created_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
