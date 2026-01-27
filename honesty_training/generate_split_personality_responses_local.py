"""
Generate split personality responses using a finetuned Qwen3 model with vllm.

This script uses the balanced Q/A dataset and prompts the model to generate
honest_persona responses starting with <split-personality-token>.
"""

import json
import argparse
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List


def load_balanced_dataset(input_path: str) -> List[dict]:
    """Load the balanced Q/A dataset."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: List[dict], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def format_split_personality_prompt(
    system_prompt: str,
    question: str,
    response: str,
) -> str:
    """Format the prompt with split personality template.

    Returns a prompt that ends with the model ready to continue from
    <split-personality-token> in the honest_persona role.
    """
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>\n"
        f"<|im_start|>honest_persona\n<split-personality-token>"
    )
    return prompt


def run_generation(
    model_path: str,
    input_path: str,
    output_path: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    response_types: List[str] = None,
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    batch_size: int = 100,
    disable_compile: bool = False,
):
    """Run the generation pipeline.

    Args:
        model_path: Path to base model.
        input_path: Path to balanced Q/A dataset.
        output_path: Path to save results.
        system_prompt: System prompt to use in the template.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        response_types: Optional list of response types to filter.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length.
        batch_size: Number of samples to process in parallel.
        disable_compile: Disable torch.compile for faster startup.
    """
    print(f"Loading model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Batch size: {batch_size}")
    if max_model_len:
        print(f"Max model length: {max_model_len}")
    if lora_adapter_path:
        print(f"Using LoRA adapter: {lora_adapter_path}")
    if disable_compile:
        print("Torch compile disabled for faster startup")

    # Initialize vllm model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enable_lora=True if lora_adapter_path else False,
        max_lora_rank=64 if lora_adapter_path else None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=disable_compile,
    )

    # Create LoRA request if adapter is specified
    lora_request = None
    if lora_adapter_path:
        lora_request = LoRARequest("split_personality_adapter", 1, lora_adapter_path)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    print(f"\nLoading balanced dataset from: {input_path}")
    data = load_balanced_dataset(input_path)
    print(f"Loaded {len(data)} samples")

    # Filter by response type if specified
    if response_types:
        data = [d for d in data if d.get("response_type") in response_types]
        print(f"Filtered to {len(data)} samples with types: {response_types}")

    if not data:
        print("No samples to evaluate. Exiting.")
        return

    # Print distribution
    type_counts = {}
    for item in data:
        rt = item.get("response_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"\nResponse type distribution:")
    for rt, count in sorted(type_counts.items()):
        print(f"  {rt}: {count}")

    print(f"\nSystem prompt: {system_prompt}")
    print(f"\nTemperature: {temperature}")
    print(f"Max tokens: {max_tokens}")

    # Process in batches
    results = []

    for batch_start in range(0, len(data), batch_size):
        batch = data[batch_start:batch_start + batch_size]

        print(f"\nProcessing batch {batch_start // batch_size + 1}/"
              f"{(len(data) + batch_size - 1) // batch_size}")

        # Prepare prompts for batch
        prompts = []
        for item in batch:
            prompt = format_split_personality_prompt(
                system_prompt,
                item["question"],
                item["response_text"],
            )
            prompts.append(prompt)

        # Generate responses in batch
        try:
            if lora_request:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(prompts, sampling_params)

            # Process outputs
            for item, output in zip(batch, outputs):
                response_content = output.outputs[0].text

                result = {
                    **item,
                    "split_personality_response": {
                        "model": model_path,
                        "lora_adapter": lora_adapter_path,
                        "system_prompt": system_prompt,
                        "honest_persona_continuation": response_content,
                    }
                }
                results.append(result)

        except Exception as e:
            print(f"    Error processing batch: {e}")
            # Add error entries for this batch
            for item in batch:
                result = {
                    **item,
                    "split_personality_response": {
                        "model": model_path,
                        "lora_adapter": lora_adapter_path,
                        "system_prompt": system_prompt,
                        "honest_persona_continuation": None,
                        "error": str(e),
                    }
                }
                results.append(result)

        save_results(results, output_path)
        print(f"  Progress: {len(results)}/{len(data)} generated")

    print(f"\nGeneration complete! Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {len(results)}")

    successful = sum(1 for r in results if r["split_personality_response"]["honest_persona_continuation"])
    errors = len(results) - successful
    print(f"Successful generations: {successful}")
    print(f"Errors: {errors}")

    # Summary by response type
    print("\nBy response type:")
    for rt in ["correct", "partial", "evasive", "lie"]:
        rt_results = [r for r in results if r.get("response_type") == rt]
        if rt_results:
            print(f"  {rt}: {len(rt_results)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate split personality responses using finetuned Qwen3 model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to local model or HuggingFace model ID",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/data/balanced_qa_dataset.json",
        help="Path to balanced Q/A dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="honesty_training/results/split_personality_responses_local.json",
        help="Path to save generation results",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to use in the conversation template",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--response-types",
        type=str,
        nargs="+",
        choices=["correct", "partial", "evasive", "lie"],
        default=None,
        help="Filter to specific response types (default: all)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory to use (0-1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of samples to process in parallel.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup but slightly slower inference.",
    )

    args = parser.parse_args()

    run_generation(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        response_types=args.response_types,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        disable_compile=args.disable_compile,
    )


if __name__ == "__main__":
    main()
