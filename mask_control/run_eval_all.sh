#!/bin/bash

source /root/.venv/bin/activate

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/split-personality-repo-epoch-4

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-split-personality-a-prompt

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-split-personality

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-honesty-finetuned-goals

python mask_control/eval_mask.py --inference-mode local --lora-adapter hcasademunt/qwen3-32b-honesty-finetuned-mixed
