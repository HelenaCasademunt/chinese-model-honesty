#!/bin/bash

# Training script for Qwen3-32B with:
# - Split personality training
# - On-policy (data from same model)
# - M9 (100% exclude system prompts)
# - No LoRA patch
# - hp elicitation

cd /root/SplitPersonalityTraining

echo "=========================================="
echo "Step 1: Setup Data Symlink"
echo "=========================================="

# Create symlink to existing stage_3_tagged data
mkdir -p training/data
if [ -L training/data/claude_data ]; then
    echo "Symlink already exists, removing old one..."
    rm training/data/claude_data
fi

ln -s ../../data/stage_3_tagged training/data/claude_data
echo "Created symlink: training/data/claude_data -> ../data/stage_3_tagged"
echo "Using existing qwen/qwen3-32b data (no need to regenerate)"

echo ""
echo "=========================================="
echo "Step 2: Create Training Config"
echo "=========================================="

# Create config file with all required settings
cat > training/cfg.json << 'EOF'
{
    "target_model": "Qwen/Qwen3-32B",
    "data_model_source": "qwen/qwen3-32b",

    "elicitation_type": "hp",
    "add_sp_token": 1,

    "system_tag": "<SYSTEM> ",
    "intervention_prefix": " INTERVENTION: ",
    "review_prefix": "\nREVIEW:",
    "flag_prefix": "\nFLAG: ",

    "intervention_types": [
        "split_personality__version_3",
        "split_personality__version_4"
    ],

    "train_topics": [
        "confidence_assessment",
        "jailbreak_attempts",
        "oversimplification",
        "fabricated_statistics",
        "goal_misgeneralization",
        "reward_hacks",
        "specification_gaming",
        "sycophancy",
        "unethical_instructions",
        "influence_seeking",
        "malicious_user_queries"
    ],

    "validation_topics": [],
    "val_samples_per_topic": 128,

    "tags_to_filter": [
        "confidence_below_4",
        "intervention_tags_disagree_once",
        "intervention_tags_disagree_twice",
        "mismatching_review_focus",
        "missing_consensus_on_b",
        "missing_consensus_on_a",
        "unclear_a_b_separation",
        "generation_exceeded_token_limit",
        "detection_difficulty_easier_externally_than_internally",
        "training_data_leakage",
        "failed_external_review"
    ],

    "prob_exclude_system_prompt": 1.0,
    "prob_mismatch_prompts": 0.0,
    "lora_patch": false,

    "seq_len": 2048,
    "batch_size": 16,
    "num_samples": -1,
    "epochs": 4,

    "claude_frac": 1.0,

    "log_every": 32,
    "use_dummy_data": 0,
    "validation_batch_size": 1,

    "lora_r": 64,
    "lora_dropout": 0.0,

    "lr": 0.00001,
    "lr_warmup_frac": 0.1,
    "lr_cooldown_frac": 0.5,
    "beta1": 0.9,
    "beta2": 0.99,
    "max_grad_norm": 1.0,
    "weight_decay": 0.0,
    "eps": 1e-8
}
EOF

echo "Config created at training/cfg.json"
echo ""
echo "Training settings:"
echo "  - Model: qwen/qwen3-32b (on-policy)"
echo "  - System prompt exclusion: 100% (M9)"
echo "  - LoRA patch: disabled"
echo "  - Elicitation: hp"
echo "  - Intervention: split_personality v3, v4"
echo ""

echo "=========================================="
echo "Step 3: Run Training"
echo "=========================================="

cd training

# Single GPU training
python train_lora.py

# For multi-GPU, use instead:
# torchrun --nproc-per-node=<NUM_GPUS> train_lora.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
