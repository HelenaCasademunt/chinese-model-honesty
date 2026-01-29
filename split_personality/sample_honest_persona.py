"""
Sample only the honest persona using pre-existing responses from Q/A datasets.

Supports two dataset formats:
- balanced: balanced_qa_dataset.json (uses response_text field)
- responses: split_personality_responses.json (uses initial_raw from model_responses)

Uses the split personality template with:
1. User question from dataset
2. Assistant response from dataset (pre-existing)
3. Honest persona continuation (newly sampled)
"""

import json
import argparse
import os
import re
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List


def load_balanced_dataset(input_path: str, data_format: str = "balanced") -> List[dict]:
    """Load the Q/A dataset and normalize it to a common format.

    Args:
        input_path: Path to the dataset JSON file.
        data_format: Either "balanced" or "responses".
            - "balanced": expects response_text field
            - "responses": expects model_responses list with initial_raw

    Returns:
        List of normalized dataset items with response_text field.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data_format == "responses":
        # Convert responses format to normalized format
        normalized = []
        for item in data:
            if "model_responses" not in item or not item["model_responses"]:
                continue

            # Create one entry per model_response
            for model_resp in item["model_responses"]:
                # Clean response text by removing split personality tokens and flags
                response_text = model_resp["initial_raw"]
                response_text = response_text.replace("<split-personality-token>", "")
                response_text = response_text.replace(" FLAG: PASS", "")
                response_text = response_text.replace(" FLAG: FAIL", "")

                normalized_item = {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "topic": item["topic"],
                    "subtopic": item.get("subtopic"),
                    "level": item.get("level"),
                    "response_text": response_text,
                    "response_type": None,  # Not available in responses format
                    "source": "split_personality_responses",
                }
                normalized.append(normalized_item)
        return normalized
    elif data_format == "balanced":
        # Already in the correct format
        return data
    else:
        raise ValueError(f"Unknown data_format: {data_format}. Must be 'balanced' or 'responses'.")


def save_results(results: List[dict], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def parse_response(content: str) -> dict:
    """Separate thinking from final answer.

    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    This function extracts the thinking portion and the final answer separately.
    """
    if content is None:
        return {"thinking": None, "answer": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    # Remove thinking tags to get the final answer
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


def format_split_personality_prompt(
    question: str,
    assistant_response: str,
    system_prompt: str = None,
    template: str = "honest_persona",
    intervention: str = "",
) -> str:
    """Format the prompt with split personality template.

    Templates:
        honest_persona: Ends with an honest_persona role after the assistant response.
        original_repo: Places <split-personality-token> inline at end of assistant response,
            optionally followed by an intervention string.
    """
    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n"

    if template == "honest_persona":
        prompt += f"<|im_start|>assistant\n{assistant_response}<|im_end|>\n"
        prompt += f"<|im_start|>honest_persona\n<split-personality-token>{intervention}"
    elif template == "original_repo":
        prompt += f"<|im_start|>assistant\n{assistant_response}<split-personality-token>{intervention}"
    else:
        raise ValueError(f"Unknown template: {template}")
    return prompt


def has_valid_responses(result: dict) -> bool:
    """Check if a result has all valid (non-null) responses."""
    responses = result.get("honest_responses", [])
    if not responses:
        return False
    return all(r.get("honest_assessment") is not None for r in responses)


def load_existing_results(output_path: str, mode: str = "skip") -> tuple:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess questions with errors/null answers,
              "overwrite" to reprocess all questions.

    Returns (results_list, set_of_completed_question_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Only consider questions complete if all responses are valid
        completed_ids = {r["question_id"] for r in results if has_valid_responses(r)}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching question_id."""
    results_by_id = {r["question_id"]: r for r in existing}
    for r in new_results:
        results_by_id[r["question_id"]] = r
    return list(results_by_id.values())


def run_evaluation(
    model_path: str,
    input_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int = 2048,
    system_prompt: str = None,
    response_types: List[str] = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    batch_size: int = 10,
    disable_compile: bool = False,
    template: str = "honest_persona",
    intervention: str = "",
    data_format: str = "balanced",
):
    """Run evaluation sampling only honest persona using balanced dataset responses.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        input_path: Path to balanced Q/A dataset.
        output_path: Path to save results.
        temperature: Sampling temperature.
        num_samples: Number of honest persona samples per question.
        max_tokens: Max tokens for honest persona response.
        system_prompt: Optional system prompt.
        response_types: Filter to specific response types (only works with balanced format).
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
        disable_compile: Disable torch.compile for faster startup.
        template: Prompt template to use (honest_persona or original_repo).
        intervention: Optional text appended after <split-personality-token> in original_repo template.
        data_format: Either "balanced" or "responses" to indicate dataset format.
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
    print(f"Prompt template: {template}")

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

    # Sampling parameters for honest persona
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
        stop=["<|im_end|>"],
    )

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    print(f"Mode: {mode}")
    print(f"Data format: {data_format}")

    print(f"\nLoading dataset from: {input_path}")
    data = load_balanced_dataset(input_path, data_format=data_format)
    print(f"Loaded {len(data)} samples")

    # Filter by response type if specified (only works with balanced format)
    if response_types:
        if data_format == "responses":
            print(f"Warning: --response-types filter not supported for 'responses' format, ignoring")
        else:
            data = [d for d in data if d.get("response_type") in response_types]
            print(f"Filtered to {len(data)} samples with types: {response_types}")

    if not data:
        print("No samples to evaluate. Exiting.")
        return []

    # Print distribution
    type_counts = {}
    for item in data:
        rt = item.get("response_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"\nResponse type distribution:")
    for rt, count in sorted(type_counts.items()):
        print(f"  {rt}: {count}")

    print(f"\nTemperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Samples per question: {num_samples}")

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")

    # Filter out already completed questions
    remaining = [d for d in data if d["question_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

    overall_start = time.time()

    # Process in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")
        print(f"Questions {batch_start + 1}-{batch_end} of {len(remaining)}")
        print(f"{'='*60}")

        # Prepare prompts for honest persona sampling
        prompts = []
        for item in batch:
            prompt = format_split_personality_prompt(
                item["question"],
                item["response_text"],
                system_prompt,
                template=template,
                intervention=intervention,
            )
            prompts.append(prompt)

        batch_start_time = time.time()
        try:
            print("  Generating honest persona assessments...")
            if lora_request:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(prompts, sampling_params)

            # Process outputs
            batch_results = []
            for idx, (item, output) in enumerate(zip(batch, outputs)):
                honest_responses = []
                for completion in output.outputs:
                    honest_response = completion.text
                    honest_parsed = parse_response(honest_response)

                    honest_responses.append({
                        "honest_raw": honest_response,
                        "honest_thinking": honest_parsed["thinking"],
                        "honest_assessment": honest_parsed["answer"],
                    })

                result = {
                    **item,
                    "honest_responses": honest_responses,
                }
                batch_results.append(result)

                topic_info = item.get("topic", "unknown")
                if item.get("subtopic"):
                    topic_info += f" > {item['subtopic']}"
                if item.get("level"):
                    topic_info += f" [{item['level']}]"

                valid_count = len([r for r in honest_responses if r['honest_assessment']])
                print(f"    [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} complete")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            # Save progress after each batch
            results = merge_results(results, batch_results)
            save_results(results, output_path)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Fall back to processing one at a time for this batch
            print("  Retrying questions individually...")
            for idx, item in enumerate(batch):
                try:
                    prompt = format_split_personality_prompt(
                        item["question"],
                        item["response_text"],
                        system_prompt,
                        template=template,
                        intervention=intervention,
                    )

                    if lora_request:
                        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
                    else:
                        outputs = llm.generate([prompt], sampling_params)

                    honest_responses = []
                    for completion in outputs[0].outputs:
                        honest_response = completion.text
                        honest_parsed = parse_response(honest_response)

                        honest_responses.append({
                            "honest_raw": honest_response,
                            "honest_thinking": honest_parsed["thinking"],
                            "honest_assessment": honest_parsed["answer"],
                        })

                    result = {
                        **item,
                        "honest_responses": honest_responses,
                    }

                    results = merge_results(results, [result])
                    save_results(results, output_path)
                    print(f"  [{batch_start + idx + 1}] ✓ Completed individually")

                except Exception as e2:
                    print(f"  [{batch_start + idx + 1}] ⚠ Failed: {type(e2).__name__}")
                    continue

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✓ ALL COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Progress: {len(results)}/{len(data)} questions complete")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    # Print summary by response type
    print("\nSummary by response type:")
    for rt in ["correct", "partial", "evasive", "lie"]:
        rt_results = [r for r in results if r.get("response_type") == rt]
        if rt_results:
            successful = sum(1 for r in rt_results if has_valid_responses(r))
            print(f"  {rt}: {len(rt_results)} samples ({successful} with valid responses)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sample only honest persona using balanced dataset responses"
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
        "--input",
        type=str,
        default="results/data/balanced_qa_dataset.json",
        help="Path to balanced Q/A dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="split_personality/results/honest_persona_balanced.json",
        help="Path to save collected responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of honest persona samples per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for honest persona response",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to use",
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
        "--mode",
        type=str,
        default="skip",
        choices=["skip", "overwrite"],
        help="skip: only process incomplete questions, overwrite: reprocess all",
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
        default=0.95,
        help="Fraction of GPU memory to use (0-1). Increase if you have OOM errors.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model. Reduce if you have KV cache memory errors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to process in parallel.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup (2min faster) but slightly slower inference.",
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["honest_persona", "original_repo"],
        default="honest_persona",
        help="Prompt template to use (default: honest_persona)",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default="",
        help="Optional text appended after <split-personality-token> in original_repo template",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["balanced", "responses"],
        default="balanced",
        help="Dataset format: 'balanced' for balanced_qa_dataset.json, 'responses' for split_personality_responses.json",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        response_types=args.response_types,
        mode=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        disable_compile=args.disable_compile,
        template=args.template,
        intervention=args.intervention,
        data_format=args.data_format,
    )


if __name__ == "__main__":
    main()
