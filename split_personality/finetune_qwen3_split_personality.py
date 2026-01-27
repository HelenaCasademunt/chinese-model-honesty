"""
Finetune Qwen3 32B with LoRA for split personality assessment using Unsloth.
Trains on data with system, user, assistant, and honest_persona roles.
"""

import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, login
import torch


def preprocess_dataset_with_masking(dataset, tokenizer, max_length):
    """
    Preprocess dataset to tokenize and add masked labels.
    Masks everything up to and including <split-personality-token>.
    """
    def process_example(example):
        text = example["text"]

        # Tokenize the full text
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,  # Don't pad yet, will pad in collator
            return_tensors=None,
        )

        # Create labels with masking
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # Find where the actual honest content starts (after <split-personality-token>)
        token_marker = "<split-personality-token>"
        marker_pos = text.find(token_marker)

        if marker_pos == -1:
            # No marker found - mask everything
            labels = [-100] * len(input_ids)
        else:
            # Find position after the marker token
            text_after_marker_start = marker_pos + len(token_marker)
            text_before_content = text[:text_after_marker_start]

            # Tokenize up to and including the marker to find token position
            tokens_before = tokenizer(
                text_before_content,
                truncation=False,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            mask_until = len(tokens_before)

            # Mask everything before the actual content
            for i in range(min(mask_until, len(labels))):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    print("Preprocessing dataset with masking...")
    processed = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing and masking",
    )
    return processed


class DataCollatorForMaskedTraining:
    """
    Simple data collator that pads pre-processed examples.
    The dataset is already tokenized with masked labels.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # Pad the batch
        import torch

        # Find max length in batch
        max_len = max(len(ex["input_ids"]) for ex in examples)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for ex in examples:
            input_ids = ex["input_ids"]
            labels = ex["labels"]

            # attention_mask might not be present depending on SFTTrainer preprocessing
            # If not present, create it (1 for all real tokens)
            if "attention_mask" in ex:
                attention_mask = ex["attention_mask"]
            else:
                attention_mask = [1] * len(input_ids)

            # Pad to max length
            padding_length = max_len - len(input_ids)

            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # Pad labels with -100

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3 32B with split personality data")
    parser.add_argument("--data", type=str, default="split_personality/data/split_personality_qwen3.jsonl", help="Path to formatted JSONL data")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--output-dir", type=str, default="/workspace/qwen3-32b-split-personality", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=3072, help="Maximum sequence length")
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

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Loaded {len(dataset)} examples")

    # Optionally limit samples
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
        print(f"Using {len(dataset)} examples for training")

    # Preprocess dataset: tokenize and add masked labels
    dataset = preprocess_dataset_with_masking(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    # Create simple collator for padding
    data_collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)

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

    # Create trainer
    # Dataset is already tokenized and has masked labels, so we don't need
    # dataset_text_field or formatting_func
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        max_seq_length=args.max_seq_length,
        packing=False,  # Can't pack when using pre-masked labels
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
