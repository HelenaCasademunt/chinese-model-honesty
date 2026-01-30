#!/bin/bash

source /root/.venv/bin/activate

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/split-personality-repo-epoch-4 --output mask_control/results/mask_eval_split_personality_repo_epoch_4.json

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-split-personality-a-prompt --output mask_control/results/mask_eval_split_personality_a_prompt.json

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-split-personality --output mask_control/results/mask_eval_split_personality.json

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-honesty-finetuned-goals --output mask_control/results/mask_eval_honesty_finetuned_goals.json

