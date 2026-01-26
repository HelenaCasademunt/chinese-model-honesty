"""
Finetune Qwen3 32B with LoRA using Unsloth for efficient single H100 training.
Trains directly on the "text" field from the JSONL without additional chat formatting.
"""

import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3 32B with LoRA")
    parser.add_argument("--data", type=str, default="goals-data-qwen3.jsonl", help="Path to training data JSONL")
    parser.add_argument("--output-dir", type=str, default="./qwen3-32b-lora-finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Load model with unsloth optimizations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-32B",
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect (will use bfloat16 on H100)
        load_in_4bit=True,  # 4-bit quantization to fit in H100
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
        lora_dropout=0.0,  # Optimized for unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Loaded {len(dataset)} training examples")

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,  # Use bfloat16 on H100
        logging_steps=10,
        optim="adamw_8bit",  # Memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="cosine",
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

    # Save final model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save merged model for easier inference
    merged_dir = f"{args.output_dir}-merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("Training complete!")


if __name__ == "__main__":
    main()
