"""
Convert split personality intervention results to the format expected by
the chinese_auditing evaluation pipeline.

Input format (from sample_honest_persona.py):
  List of dicts with question_id, question, response_text, honest_responses, etc.

Output format (for evaluation_pipeline.py):
  {"config": {...}, "results": [{prompt_id, prompt, response, sample_idx, model, target_aspect}, ...]}
"""

import argparse
import json
from datetime import datetime, timezone


def convert(input_path, output_path, adapter_name):
    with open(input_path) as f:
        data = json.load(f)

    results = []
    for item in data:
        question_id = item.get("question_id", "")
        question = item.get("question", "")
        topic = item.get("topic", "")
        subtopic = item.get("subtopic", "")
        level = item.get("level", "")
        sample_idx = item.get("sample_idx", 0)

        # Build target_aspect matching the convention: level/topic/subtopic
        if topic:
            target_aspect = f"{level or 'unknown'}/{topic}/{subtopic or 'unknown'}"
        else:
            target_aspect = ""

        for resp in item.get("honest_responses", []):
            honest_assessment = resp.get("honest_assessment", "")
            if not honest_assessment:
                continue

            results.append({
                "prompt_id": str(question_id),
                "sample_idx": sample_idx if isinstance(sample_idx, int) else 0,
                "prompt": question,
                "response": honest_assessment,
                "model": adapter_name,
                "target_aspect": target_aspect,
            })

    output = {
        "config": {
            "model": adapter_name,
            "source": str(input_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(results)} responses -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert intervention results for evaluation pipeline")
    parser.add_argument("--input", required=True, help="Path to intervention result JSON")
    parser.add_argument("--output", required=True, help="Path to save converted JSON")
    parser.add_argument("--adapter-name", required=True, help="Name of the LoRA adapter")
    args = parser.parse_args()

    convert(args.input, args.output, args.adapter_name)


if __name__ == "__main__":
    main()
