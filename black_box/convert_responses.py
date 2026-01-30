"""
Convert black_box response files to chinese_auditing format.

black_box scripts output responses in several formats:
  - baseline/system_prompt: list of {question_id, topic, subtopic, level, question, model_responses: [{raw, thinking, answer}]}
  - assistant_prefill/pretrain: dict of {topic: [{question_id, ..., model_responses: [str]}]}
  - user_prefill: list of {question_id, ..., continuations: [str], full_user_messages: [str]}

chinese_auditing expects:
  {config: {...}, results: [{prompt_id, prompt, target_aspect, sample_idx, model, response}]}

Each sample is a separate entry in results (one per response, not grouped by question).
"""

import json
import argparse
import os
from pathlib import Path


def extract_response_text(resp):
    """Extract the text from a response, handling both str and dict formats."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        # Prefer 'answer' (strips thinking), fall back to 'raw'
        return resp.get("answer") or resp.get("raw") or ""
    return ""


def convert_list_format(data, model_name):
    """Convert list-of-dicts format (baseline, system_prompt, lora evals)."""
    results = []
    prompt_counter = 0
    for item in data:
        topic = item.get("topic", "")
        subtopic = item.get("subtopic", "")
        level = item.get("level", "")
        target_aspect = f"{topic}/{subtopic}/{level}" if subtopic else f"{topic}/{level}"
        question = item.get("question", "")

        responses = item.get("model_responses", [])
        for sample_idx, resp in enumerate(responses):
            prompt_counter += 1
            results.append({
                "prompt_id": str(prompt_counter),
                "prompt": question,
                "formatted_prompt": "",
                "target_aspect": target_aspect,
                "sample_idx": sample_idx,
                "model": model_name,
                "response": extract_response_text(resp),
            })
    return results


def convert_dict_format(data, model_name):
    """Convert dict-keyed format (assistant_prefill, pretrain_prompts)."""
    results = []
    prompt_counter = 0
    for topic_key, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            topic = item.get("topic", topic_key)
            subtopic = item.get("subtopic", "")
            level = item.get("level", "")
            target_aspect = f"{topic}/{subtopic}/{level}" if subtopic else f"{topic}/{level}"
            question = item.get("question", "")

            responses = item.get("model_responses", [])
            for sample_idx, resp in enumerate(responses):
                prompt_counter += 1
                results.append({
                    "prompt_id": str(prompt_counter),
                    "prompt": question,
                    "formatted_prompt": "",
                    "target_aspect": target_aspect,
                    "sample_idx": sample_idx,
                    "model": model_name,
                    "response": extract_response_text(resp),
                })
    return results


def convert_user_prefill_format(data, model_name):
    """Convert user prefill format (continuations instead of model_responses)."""
    results = []
    prompt_counter = 0
    for item in data:
        topic = item.get("topic", "")
        subtopic = item.get("subtopic", "")
        level = item.get("level", "")
        target_aspect = f"{topic}/{subtopic}/{level}" if subtopic else f"{topic}/{level}"
        question = item.get("question", "")

        # Use full_user_messages if available, else continuations
        responses = item.get("full_user_messages", item.get("continuations", []))
        for sample_idx, resp in enumerate(responses):
            prompt_counter += 1
            results.append({
                "prompt_id": str(prompt_counter),
                "prompt": question,
                "formatted_prompt": "",
                "target_aspect": target_aspect,
                "sample_idx": sample_idx,
                "model": model_name,
                "response": extract_response_text(resp),
            })
    return results


def detect_and_convert(data, model_name):
    """Auto-detect format and convert."""
    if isinstance(data, list):
        # List format: check if it has continuations (user_prefill) or model_responses
        if data and "continuations" in data[0]:
            return convert_user_prefill_format(data, model_name)
        return convert_list_format(data, model_name)
    elif isinstance(data, dict):
        # Dict format: could be topic-keyed (assistant_prefill, pretrain)
        return convert_dict_format(data, model_name)
    return []


def convert_file(input_path, output_path, model_name):
    """Convert a single file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = detect_and_convert(data, model_name)

    output = {
        "config": {
            "model": model_name,
            "source_file": str(input_path),
            "n_samples": max((r["sample_idx"] for r in results), default=0) + 1 if results else 0,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({len(results)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert black_box response files to chinese_auditing format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results_new_pipeline/data",
        help="Directory containing black_box response files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_new_pipeline/converted",
        help="Directory to write converted files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-32b",
        help="Model name to set in converted output",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting files from {input_dir} -> {output_dir}")
    print(f"Model: {args.model}")
    print()

    # Convert top-level JSON files
    for json_file in sorted(input_dir.glob("*.json")):
        out_name = json_file.name
        convert_file(str(json_file), str(output_dir / out_name), args.model)

    # Convert subdirectory files (system_prompts/, pretrain_prompts/)
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            sub_output = output_dir / subdir.name
            os.makedirs(sub_output, exist_ok=True)
            for json_file in sorted(subdir.glob("*.json")):
                convert_file(
                    str(json_file),
                    str(sub_output / json_file.name),
                    args.model,
                )

    print(f"\nDone. Converted files in: {output_dir}")


if __name__ == "__main__":
    main()
