"""
Finetune Qwen3 32B with LoRA using Unsloth for efficient single H100 training.
Trains directly on the "text" field from the JSONL without additional chat formatting.
Supports training on goals dataset, followup dataset, or both mixed 50/50.
"""

import argparse
from datasets import load_dataset, interleave_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3 32B with LoRA using mixed datasets")
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
    args = parser.parse_args()

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

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
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

    # Create trainer - train directly on "text" field
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,  # Pack sequences for efficiency
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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
