"""
Local baseline evaluation script using vllm.
Reads questions from dev_questions.json format and outputs in the same format as qwen_baseline_responses.
"""

import json
import argparse
import os
import time
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_questions(json_path: str) -> list:
    """Load questions from JSON file.

    Supports two formats:
    1. Array format: [{"question": ..., "topic": ..., "level": ..., "prompt_id": ...}]
    2. Nested format with categories

    Returns a flat list of question dicts with normalized fields.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    if isinstance(data, list):
        # Array format like dev_questions.json
        for q in data:
            questions.append({
                "prompt_id": q.get("prompt_id", ""),
                "question": q["question"],
                "topic": q.get("topic", "unknown"),
                "level": q.get("level", "unknown"),
            })
    else:
        # Nested format with categories
        for category_data in data.get("categories", []):
            category = category_data.get("name", "unknown")
            for q in category_data.get("questions", []):
                questions.append({
                    "prompt_id": q.get("prompt_id", ""),
                    "question": q["question"],
                    "topic": category,
                    "level": q.get("level", "unknown"),
                })

    return questions


def format_prompt(question: str, chat_template: str = "qwen3", enable_reasoning: bool = False) -> str:
    """Format the prompt using the specified chat template."""
    if chat_template == "qwen3":
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        if enable_reasoning:
            prompt += "<think>\n\n</think>\n\n"
        return prompt
    else:
        # Default format
        return f"User: {question}\n\nAssistant:"


def save_results(results: dict, output_path: str):
    """Save results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_evaluation(
    model_path: str,
    questions_path: str,
    output_path: str,
    temperature: float = 1.0,
    num_samples: int = 10,
    max_tokens: int = 10000,
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    batch_size: int = 10,
    max_concurrent: int = 20,
    chat_template: str = "qwen3",
    enable_reasoning: bool = False,
    disable_compile: bool = False,
):
    """Run the full evaluation collecting multiple answers per question.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        questions_path: Path to questions JSON file.
        output_path: Path to save collected responses.
        temperature: Sampling temperature.
        num_samples: Number of responses to collect per question.
        max_tokens: Maximum tokens for model responses.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
        max_concurrent: Maximum concurrent requests (stored in config).
        chat_template: Chat template to use (qwen3, etc).
        enable_reasoning: Whether to enable reasoning mode.
        disable_compile: Disable torch.compile for faster startup.
    """
    print(f"Loading model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Batch size: {batch_size}")
    print(f"Chat template: {chat_template}")
    print(f"Enable reasoning: {enable_reasoning}")
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
        lora_request = LoRARequest("adapter", 1, lora_adapter_path)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,  # Generate multiple samples per prompt
    )

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Initialize output structure
    output_data = {
        "config": {
            "model": model_path,
            "prompts_csv": questions_path,
            "output_dir": os.path.dirname(output_path),
            "n_samples": num_samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_concurrent": max_concurrent,
            "use_chat_api": False,
            "chat_template": chat_template,
            "enable_reasoning": enable_reasoning,
            "max_retries": 10,
            "retry_delay": 2.0,
            "batch_size": batch_size,
        },
        "results": []
    }

    overall_start = time.time()

    # Process questions in batches
    for batch_start in range(0, len(questions), batch_size):
        batch_end = min(batch_start + batch_size, len(questions))
        batch = questions[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(questions) + batch_size - 1) // batch_size}")
        print(f"Questions {batch_start + 1}-{batch_end} of {len(questions)}")
        print(f"{'='*60}")

        # Format all prompts in batch
        prompts = [format_prompt(q["question"], chat_template, enable_reasoning) for q in batch]

        # Generate responses for all questions in batch
        batch_start_time = time.time()
        try:
            if lora_request:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(prompts, sampling_params)

            # Process each question's outputs
            for idx, (question, output) in enumerate(zip(batch, outputs)):
                for sample_idx, completion in enumerate(output.outputs):
                    response_text = completion.text

                    # Construct target_aspect
                    target_aspect = f"{question.get('level', 'unknown')}/{question.get('topic', 'unknown')}/unknown"

                    # Add result entry
                    result_entry = {
                        "prompt_id": question["prompt_id"],
                        "prompt": question["question"],
                        "formatted_prompt": prompts[idx],
                        "target_aspect": target_aspect,
                        "sample_idx": sample_idx,
                        "model": model_path,
                        "response": response_text,
                        "usage": {
                            "completion_tokens": len(completion.token_ids) if hasattr(completion, 'token_ids') else 0
                        }
                    }
                    output_data["results"].append(result_entry)

                print(f"  [{batch_start + idx + 1}] Prompt {question['prompt_id']}: {num_samples} samples generated")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            # Save progress after each batch
            save_results(output_data, output_path)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Continue to next batch
            continue

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✓ ALL COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Total results: {len(output_data['results'])} entries")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline responses using vllm with qwen format output"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model ID (e.g., Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/dev_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save collected responses (default: results/qwen_baseline_responses_<timestamp>.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens for model responses",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0-1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to process in parallel",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent requests (stored in config)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="qwen3",
        help="Chat template to use (qwen3, etc)",
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable reasoning mode (add empty <think> tags)",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup",
    )

    args = parser.parse_args()

    # Generate output filename with timestamp if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/qwen_baseline_responses_{timestamp}.json"

    run_evaluation(
        model_path=args.model,
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        chat_template=args.chat_template,
        enable_reasoning=args.enable_reasoning,
        disable_compile=args.disable_compile,
    )


if __name__ == "__main__":
    main()
