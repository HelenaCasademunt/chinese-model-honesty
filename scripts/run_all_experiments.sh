#!/bin/bash

source ~/.bashrc
source /root/.venv/bin/activate

echo ""
echo "=========================================="
echo "MEGASCRIPT: RUNNING ALL EXPERIMENTS"
echo "Started at: $(date)"
echo "=========================================="

########################################
# EXPERIMENT 1: Qwen + DeepSeek Attacks
########################################

echo ""
echo "=========================================="
echo "EXPERIMENT 1: Qwen + DeepSeek Attacks"
echo "=========================================="
bash /root/chinese-model-honesty/scripts/run_qwen_deepseek_attacks.sh
exp1_exit=$?
echo "Experiment 1 completed with exit code: $exp1_exit"

########################################
# EXPERIMENT 2: Qwen VL 8B Honesty Training
########################################

echo ""
echo "=========================================="
echo "EXPERIMENT 2: Qwen VL 8B Honesty Training"
echo "=========================================="
bash /root/chinese-model-honesty/scripts/run_qwen_vl_8b_honesty_qwen_vl_8b_data.sh
exp2_exit=$?
echo "Experiment 2 completed with exit code: $exp2_exit"

########################################
# EXPERIMENT 3: Qwen VL 8B Split Personality
########################################

echo ""
echo "=========================================="
echo "EXPERIMENT 3: Qwen VL 8B Split Personality"
echo "=========================================="
bash /root/chinese_auditing/scripts/run_qwen_vl_8b_split_personality.sh
exp3_exit=$?
echo "Experiment 3 completed with exit code: $exp3_exit"

########################################
# SUMMARY
########################################

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Completed at: $(date)"
echo "=========================================="
echo "Experiment 1 (Attacks):           exit code $exp1_exit"
echo "Experiment 2 (Honesty Training):  exit code $exp2_exit"
echo "Experiment 3 (Split Personality): exit code $exp3_exit"
echo "=========================================="
