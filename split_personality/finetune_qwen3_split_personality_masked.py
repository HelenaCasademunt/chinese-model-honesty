"""
Finetune Qwen3 32B with LoRA for split personality assessment using Unsloth.
Uses forward hooks to force LoRA outputs to zero for all positions before honest_persona.
This ensures the LoRA adapter only affects honest_persona generation.
"""

import argparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from typing import Optional


class LoRAMaskingCallback:
    """
    Callback that registers forward hooks on LoRA layers to mask outputs
    before the honest_persona position in each sequence.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.honest_persona_positions = None

        # Find the token ID for the honest_persona marker
        # The format is: <|im_start|>honest_persona
        test_text = "<|im_start|>honest_persona\n"
        self.honest_persona_start_token = "<|im_start|>"

    def set_honest_persona_positions(self, input_ids):
        """
        Identify the position where honest_persona starts in each sequence.
        We look for the pattern: <|im_start|>honest_persona
        """
        batch_size, seq_len = input_ids.shape
        self.honest_persona_positions = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

        # Decode each sequence to find honest_persona position
        for i in range(batch_size):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
            # Find the last occurrence of <|im_start|> followed by honest_persona
            honest_marker = "<|im_start|>honest_persona"
            honest_idx = text.rfind(honest_marker)

            if honest_idx != -1:
                # Find the token position corresponding to this character position
                # We need to tokenize up to this point
                prefix_text = text[:honest_idx]
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                self.honest_persona_positions[i] = len(prefix_tokens)
            else:
                # If not found, set to sequence length (no masking)
                self.honest_persona_positions[i] = seq_len

    def create_lora_mask_hook(self, layer_name):
        """
        Create a forward hook that masks LoRA outputs before honest_persona position.
        """
        def hook_fn(module, input, output):
            if self.honest_persona_positions is None:
                return output

            # Output shape is typically (batch_size, seq_len, hidden_dim)
            if len(output.shape) == 3:
                batch_size, seq_len, hidden_dim = output.shape

                # Create a mask: 1 for positions >= honest_persona_pos, 0 otherwise
                mask = torch.zeros(batch_size, seq_len, 1, device=output.device, dtype=output.dtype)
                for i in range(batch_size):
                    honest_pos = self.honest_persona_positions[i]
                    mask[i, honest_pos:, :] = 1.0

                # Apply mask to zero out LoRA contributions before honest_persona
                output = output * mask

            return output

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all LoRA layers."""
        self.remove_hooks()  # Clear any existing hooks

        for name, module in self.model.named_modules():
            # Check if this is a LoRA layer (contains 'lora' in name)
            if 'lora' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self.create_lora_mask_hook(name))
                self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} LoRA masking hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class MaskedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that masks LoRA outputs before honest_persona positions.
    """
    def __init__(self, *args, lora_masking_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_masking_callback = lora_masking_callback

        if self.lora_masking_callback:
            # Register hooks after model is initialized
            self.lora_masking_callback.register_hooks()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to set honest_persona positions before forward pass.
        """
        # Extract input_ids and find honest_persona positions
        if 'input_ids' in inputs and self.lora_masking_callback:
            self.lora_masking_callback.set_honest_persona_positions(inputs['input_ids'])

        # Call parent compute_loss
        return super().compute_loss(model, inputs, return_outputs)


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen3 32B with masked LoRA for split personality")
    parser.add_argument("--data", type=str, default="split_personality/data/split_personality_qwen3.jsonl", help="Path to formatted JSONL data")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--output-dir", type=str, default="/workspace/qwen3-32b-split-personality-lora-masked", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Load model with unsloth optimizations
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-32B",
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect (will use bfloat16 on H100)
        load_in_4bit=True,  # Required to fit 32B model in single H100
    )

    # Apply LoRA adapters
    print("Applying LoRA adapters...")
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

    # Create LoRA masking callback
    print("Setting up LoRA masking (zero outputs before honest_persona)...")
    lora_masking_callback = LoRAMaskingCallback(model, tokenizer)

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Loaded {len(dataset)} examples")

    # Optionally limit samples
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
        print(f"Using {len(dataset)} examples for training")

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

    # Create masked trainer
    print("Creating masked SFT trainer...")
    trainer = MaskedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,  # Pack sequences for efficiency
        args=training_args,
        lora_masking_callback=lora_masking_callback,
    )

    # Train
    print("Starting training with LoRA masking...")
    print("LoRA will only affect positions from honest_persona onwards")
    trainer.train()

    # Remove hooks before saving
    lora_masking_callback.remove_hooks()

    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
