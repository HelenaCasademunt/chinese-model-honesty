"""
Evaluation script for split personality models using vllm.
Performs two-stage sampling:
1. Generate initial assistant response
2. Append honest_persona and <split-personality-token>, then continue sampling
"""

import json
import argparse
import os
import re
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_questions(json_path: str) -> list:
    """Load questions from the evaluation JSON file.

    Supports two formats:
    1. Simple format: {"category": [{"question_id": ..., "question": ..., "answer": ...}]}
    2. Finegrained format: {"metadata": ..., "topic": {"subtopic": [{"level": ..., "question": ..., "expected_answer": ...}]}}

    Returns a flat list of question dicts with normalized fields.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    # Check if it's the finegrained format (has nested subtopics with "level" field)
    is_finegrained = False
    for key, value in data.items():
        if key == "metadata":
            continue
        if isinstance(value, dict):
            # Check if it has subtopics (dicts containing lists with "level")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list) and subvalue and "level" in subvalue[0]:
                    is_finegrained = True
                    break
        if is_finegrained:
            break

    if is_finegrained:
        # Finegrained format: topic > subtopic > questions
        for topic, subtopics in data.items():
            if topic == "metadata":
                continue
            for subtopic, question_list in subtopics.items():
                for i, q in enumerate(question_list):
                    question_id = f"{topic}_{subtopic}_{q.get('level', i)}"
                    questions.append({
                        "question_id": question_id,
                        "topic": topic,
                        "subtopic": subtopic,
                        "level": q.get("level"),
                        "question": q["question"],
                        "reference_answer": q.get("expected_answer", q.get("answer", "")),
                    })
    else:
        # Simple format: category > questions
        for category, question_list in data.items():
            for q in question_list:
                questions.append({
                    "question_id": q.get("question_id", ""),
                    "topic": category,
                    "subtopic": None,
                    "level": None,
                    "question": q["question"],
                    "reference_answer": q.get("answer", q.get("expected_answer", "")),
                })

    return questions


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


def format_prompt(question: str, system_prompt: str = None) -> str:
    """Format the prompt using Qwen's chat template format."""
    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def format_continuation_prompt(initial_response: str) -> str:
    """Format the continuation prompt for honest persona sampling.

    Takes the initial response (which ends with assistant content) and appends
    the honest_persona role with split-personality-token.
    """
    continuation = initial_response
    # Remove trailing whitespace/newline if any
    continuation = continuation.rstrip()
    # Add end token for assistant, then start honest_persona
    continuation += "<|im_end|>\n<|im_start|>honest_persona\n<split-personality-token>"
    return continuation


def has_valid_responses(result: dict) -> bool:
    """Check if a result has all valid (non-null) responses."""
    responses = result.get("model_responses", [])
    if not responses:
        return False
    return all(
        r.get("initial_answer") is not None and r.get("honest_assessment") is not None
        for r in responses
    )


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


def save_results(results: list, output_path: str):
    """Save results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching question_id."""
    results_by_id = {r["question_id"]: r for r in existing}
    for r in new_results:
        results_by_id[r["question_id"]] = r
    return list(results_by_id.values())


def run_evaluation(
    model_path: str,
    questions_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens_initial: int = 3072,
    max_tokens_honest: int = 2048,
    system_prompt: str = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    batch_size: int = 1,
    disable_compile: bool = False,
):
    """Run the split personality evaluation with two-stage sampling.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        max_tokens_initial: Max tokens for initial assistant response.
        max_tokens_honest: Max tokens for honest persona continuation.
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use for the model (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
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

    # Sampling parameters for initial response
    sampling_params_initial = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens_initial,
        n=num_samples,
        stop=["<|im_end|>"],  # Stop at end of assistant message
    )

    # Sampling parameters for honest persona continuation
    sampling_params_honest = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens_honest,
        n=1,  # Only need one continuation per initial response
        stop=["<|im_end|>"],
    )

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    print(f"Mode: {mode}")

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")

    # Filter out already completed questions
    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

    overall_start = time.time()

    # Process questions in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")
        print(f"Questions {batch_start + 1}-{batch_end} of {len(remaining)}")
        print(f"{'='*60}")

        # Stage 1: Generate initial assistant responses
        prompts = [format_prompt(q["question"], system_prompt) for q in batch]

        batch_start_time = time.time()
        try:
            print("  Stage 1: Generating initial responses...")
            if lora_request:
                initial_outputs = llm.generate(prompts, sampling_params_initial, lora_request=lora_request)
            else:
                initial_outputs = llm.generate(prompts, sampling_params_initial)

            # Stage 2: For each initial response, continue with honest persona
            batch_results = []
            for idx, (question, initial_output) in enumerate(zip(batch, initial_outputs)):
                print(f"  [{batch_start + idx + 1}] Stage 2: Generating honest assessments...")

                responses = []
                for completion_idx, completion in enumerate(initial_output.outputs):
                    initial_response = completion.text
                    initial_parsed = parse_response(initial_response)

                    # Create continuation prompt
                    full_prompt = prompts[idx] + initial_response
                    continuation_prompt = format_continuation_prompt(full_prompt)

                    # Generate honest assessment
                    try:
                        if lora_request:
                            honest_output = llm.generate(
                                [continuation_prompt],
                                sampling_params_honest,
                                lora_request=lora_request
                            )
                        else:
                            honest_output = llm.generate([continuation_prompt], sampling_params_honest)

                        honest_response = honest_output[0].outputs[0].text
                        honest_parsed = parse_response(honest_response)

                        responses.append({
                            "initial_raw": initial_response,
                            "initial_thinking": initial_parsed["thinking"],
                            "initial_answer": initial_parsed["answer"],
                            "honest_raw": honest_response,
                            "honest_thinking": honest_parsed["thinking"],
                            "honest_assessment": honest_parsed["answer"],
                        })
                    except Exception as e:
                        print(f"    Warning: Failed to generate honest assessment {completion_idx + 1}: {e}")
                        responses.append({
                            "initial_raw": initial_response,
                            "initial_thinking": initial_parsed["thinking"],
                            "initial_answer": initial_parsed["answer"],
                            "honest_raw": None,
                            "honest_thinking": None,
                            "honest_assessment": None,
                        })

                result = {
                    "question_id": question["question_id"],
                    "topic": question["topic"],
                    "subtopic": question["subtopic"],
                    "level": question["level"],
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "model_responses": responses,
                }
                batch_results.append(result)

                topic_info = question["topic"]
                if question["subtopic"]:
                    topic_info += f" > {question['subtopic']}"
                if question["level"]:
                    topic_info += f" [{question['level']}]"

                valid_count = len([r for r in responses if r['initial_raw'] and r['honest_raw']])
                print(f"    {topic_info}: {valid_count}/{num_samples} complete responses")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            # Save progress after each batch
            results = merge_results(results, batch_results)
            save_results(results, output_path)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Fall back to processing one at a time for this batch
            print("  Retrying questions individually...")
            for idx, question in enumerate(batch):
                try:
                    prompt = format_prompt(question["question"], system_prompt)

                    # Stage 1: Initial response
                    if lora_request:
                        initial_outputs = llm.generate([prompt], sampling_params_initial, lora_request=lora_request)
                    else:
                        initial_outputs = llm.generate([prompt], sampling_params_initial)

                    responses = []
                    for completion in initial_outputs[0].outputs:
                        initial_response = completion.text
                        initial_parsed = parse_response(initial_response)

                        # Stage 2: Honest assessment
                        full_prompt = prompt + initial_response
                        continuation_prompt = format_continuation_prompt(full_prompt)

                        try:
                            if lora_request:
                                honest_output = llm.generate(
                                    [continuation_prompt],
                                    sampling_params_honest,
                                    lora_request=lora_request
                                )
                            else:
                                honest_output = llm.generate([continuation_prompt], sampling_params_honest)

                            honest_response = honest_output[0].outputs[0].text
                            honest_parsed = parse_response(honest_response)

                            responses.append({
                                "initial_raw": initial_response,
                                "initial_thinking": initial_parsed["thinking"],
                                "initial_answer": initial_parsed["answer"],
                                "honest_raw": honest_response,
                                "honest_thinking": honest_parsed["thinking"],
                                "honest_assessment": honest_parsed["answer"],
                            })
                        except Exception as e:
                            responses.append({
                                "initial_raw": initial_response,
                                "initial_thinking": initial_parsed["thinking"],
                                "initial_answer": initial_parsed["answer"],
                                "honest_raw": None,
                                "honest_thinking": None,
                                "honest_assessment": None,
                            })

                    result = {
                        "question_id": question["question_id"],
                        "topic": question["topic"],
                        "subtopic": question["subtopic"],
                        "level": question["level"],
                        "question": question["question"],
                        "reference_answer": question["reference_answer"],
                        "model_responses": responses,
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
    print(f"  Progress: {len(results)}/{len(questions)} questions complete")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate split personality models with two-stage sampling"
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
        default="black_box/eval/prompts/finegrained_eval_questions_filtered.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="split_personality/results/split_personality_responses.json",
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
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens-initial",
        type=int,
        default=3072,
        help="Maximum tokens for initial assistant response",
    )
    parser.add_argument(
        "--max-tokens-honest",
        type=int,
        default=2048,
        help="Maximum tokens for honest persona continuation",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to use",
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
        default=4,
        help="Number of questions to process in parallel. Lower for split personality due to two-stage sampling.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup (2min faster) but slightly slower inference.",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens_initial=args.max_tokens_initial,
        max_tokens_honest=args.max_tokens_honest,
        system_prompt=args.system_prompt,
        mode=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        disable_compile=args.disable_compile,
    )


if __name__ == "__main__":
    main()
