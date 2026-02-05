"""
Finetune Qwen3 32B with LoRA using Unsloth for efficient single H100 training.
Trains only on assistant tokens (not user/system tokens) using response templates.
Supports training on goals dataset, followup dataset, or both mixed 50/50.

Usage:
    python finetune_qwen3_32b.py config.yaml
"""

import argparse
import json
import os
import yaml
from datasets import load_dataset, interleave_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, login
import torch


class DataCollatorForCompletionOnlyLMWithTemplateExclusion(DataCollatorForLanguageModeling):
    """
    Data collator that masks all tokens except the LAST assistant response.
    Also masks the response template tokens themselves (e.g., <|im_start|>assistant\n).
    For multi-turn conversations, only the final assistant turn is trained on.
    """

    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        # Tokenize the response template to get its token IDs
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # For each example in the batch, find the response template and mask everything before and including it
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].tolist()
            labels = batch["labels"][i].clone()

            # Find where the LAST response template appears (for multi-turn conversations)
            response_start_idx = None
            for j in range(len(input_ids) - len(self.response_template_ids) + 1):
                if input_ids[j:j + len(self.response_template_ids)] == self.response_template_ids:
                    # Found the template - set response_start_idx to the position AFTER the template
                    # Don't break - keep going to find the last occurrence
                    response_start_idx = j + len(self.response_template_ids)

            if response_start_idx is not None:
                # Mask everything before and including the response template
                labels[:response_start_idx] = -100
            else:
                # If template not found, mask the entire sequence
                labels[:] = -100

            batch["labels"][i] = labels

        return batch


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3 32B with LoRA (assistant tokens only)")
    parser.add_argument("config", type=str, nargs="?", default=None, help="Path to YAML config file")
    parser.add_argument("--goals-data", type=str, default="honesty_training/data/goals_data_qwen.jsonl", help="Path to goals dataset JSONL")
    parser.add_argument("--followup-data", type=str, default="honesty_training/data/followup_data-qwen3.jsonl", help="Path to followup dataset JSONL")
    parser.add_argument("--dataset-mode", type=str, default="mixed", choices=["goals", "followup", "mixed"], help="Dataset to use: goals, followup, or mixed (50/50)")
    parser.add_argument("--num-samples", type=int, default=5000, help="Total number of samples to use (for mixed mode, splits 50/50 between datasets)")
    parser.add_argument("--output-dir", type=str, default="/workspace/qwen3-32b-lora-finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Upload to Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token (optional, uses cached token if not provided)")
    cli_args = parser.parse_args()

    # Load config from YAML if provided, otherwise use CLI args
    if cli_args.config:
        print(f"Loading config from {cli_args.config}")
        config = load_config(cli_args.config)

        # Create args namespace from config with defaults from CLI
        class Args:
            pass
        args = Args()
        args.goals_data = config.get("goals_data", cli_args.goals_data)
        args.followup_data = config.get("followup_data", cli_args.followup_data)
        args.dataset_mode = config.get("dataset_mode", cli_args.dataset_mode)
        args.num_samples = config.get("num_samples", cli_args.num_samples)
        args.output_dir = config.get("output_dir", cli_args.output_dir)
        args.epochs = config.get("epochs", cli_args.epochs)
        args.batch_size = config.get("batch_size", cli_args.batch_size)
        args.grad_accum = config.get("grad_accum", cli_args.grad_accum)
        args.lr = config.get("lr", cli_args.lr)
        args.max_seq_length = config.get("max_seq_length", cli_args.max_seq_length)
        args.lora_r = config.get("lora_r", cli_args.lora_r)
        args.lora_alpha = config.get("lora_alpha", cli_args.lora_alpha)
        args.save_steps = config.get("save_steps", cli_args.save_steps)
        args.warmup_steps = config.get("warmup_steps", 5)
        args.hf_repo_id = config.get("hf_repo_id")
        args.hf_token = config.get("hf_token")
    else:
        # Use CLI args directly, converting dashes to underscores
        args = cli_args
        args.goals_data = cli_args.goals_data
        args.followup_data = cli_args.followup_data
        args.dataset_mode = cli_args.dataset_mode
        args.warmup_steps = 5

    # Load model with unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-32B",
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect (will use bfloat16 on H100)
        load_in_4bit=True,  # Required to fit 32B model in single H100
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )

    # Load and prepare dataset based on mode
    if args.dataset_mode == "goals":
        print(f"Loading goals dataset only...")
        goals_dataset = load_dataset("json", data_files=args.goals_data, split="train")
        print(f"Loaded {len(goals_dataset)} goals examples")
        dataset = goals_dataset.select(range(min(args.num_samples, len(goals_dataset))))
        print(f"Using {len(dataset)} goals examples for training")

    elif args.dataset_mode == "followup":
        print(f"Loading followup dataset only...")
        followup_dataset = load_dataset("json", data_files=args.followup_data, split="train")
        print(f"Loaded {len(followup_dataset)} followup examples")
        dataset = followup_dataset.select(range(min(args.num_samples, len(followup_dataset))))
        print(f"Using {len(dataset)} followup examples for training")

    else:  # mixed mode
        print(f"Loading both datasets for mixing...")
        goals_dataset = load_dataset("json", data_files=args.goals_data, split="train")
        followup_dataset = load_dataset("json", data_files=args.followup_data, split="train")

        print(f"Loaded {len(goals_dataset)} goals examples")
        print(f"Loaded {len(followup_dataset)} followup examples")

        # Split num_samples 50/50 between datasets
        samples_per_dataset = args.num_samples // 2
        goals_subset = goals_dataset.select(range(min(samples_per_dataset, len(goals_dataset))))
        followup_subset = followup_dataset.select(range(min(samples_per_dataset, len(followup_dataset))))

        print(f"Using {len(goals_subset)} goals examples")
        print(f"Using {len(followup_subset)} followup examples")

        # Mix datasets 50/50 by interleaving
        dataset = interleave_datasets(
            [goals_subset, followup_subset],
            probabilities=[0.5, 0.5],
            seed=42,
            stopping_strategy="all_exhausted"
        )

        print(f"Mixed dataset contains {len(dataset)} total training examples")

    # Create data collator that only trains on assistant responses
    # For Qwen ChatML format, the response template is "<|im_start|>assistant\n"
    # This custom collator also masks the response template tokens themselves
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLMWithTemplateExclusion(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,  # Use bfloat16 on H100
        logging_steps=1,
        optim="adamw_8bit",  # Memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=42,
        report_to="none",
    )

    # Create trainer - train only on assistant tokens
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=collator,  # Use completion-only data collator
        packing=False,  # Disable packing when using response templates
        args=training_args,
    )

    # Train
    print("Starting training (gradients only on assistant tokens)...")
    trainer.train()

    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config to JSON
    training_config = {
        "goals_data": args.goals_data,
        "followup_data": args.followup_data,
        "dataset_mode": args.dataset_mode,
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Saved training config to {config_path}")

    print("Training complete!")

    # Upload to Hugging Face if repo_id provided
    if args.hf_repo_id:
        print(f"\nUploading model to Hugging Face Hub: {args.hf_repo_id}")

        # Login if token provided
        if args.hf_token:
            login(token=args.hf_token)

        # Create repository if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
            print(f"Repository {args.hf_repo_id} ready")
        except Exception as e:
            print(f"Note: {e}")

        # Upload the model
        print(f"Uploading {args.output_dir} to {args.hf_repo_id}...")
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hf_repo_id,
            repo_type="model",
        )

        print(f"✓ Successfully uploaded to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
