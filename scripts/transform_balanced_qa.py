"""
Transform balanced_qa_dataset.json into the format expected by honesty_eval/evaluate.py.

Input format (balanced_qa_dataset.json):
- Array of objects with: question_id, question, reference_answer, topic, subtopic,
  level, response_text, response_type, fact_checks, source

Output format (compatible with evaluate.py "simple" type):
- {"config": {...}, "results": [...]}
- Each result has: prompt_id, prompt, response, response_type, and other metadata
"""

import json
import argparse
from pathlib import Path


def transform_dataset(input_path: str, output_path: str, model_name: str = "unknown"):
    """Transform balanced_qa_dataset.json to evaluation format."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for idx, item in enumerate(data):
        result = {
            "prompt_id": item.get("question_id", str(idx)),
            "prompt": item.get("question", ""),
            "formatted_prompt": item.get("question", ""),
            "target_aspect": f"{item.get('topic', 'unknown')}/{item.get('subtopic', 'unknown')} [{item.get('level', 'unknown')}]",
            "sample_idx": idx,
            "model": model_name,
            "response": item.get("response_text", ""),
            "response_type": item.get("response_type", ""),
            "source": item.get("source", ""),
            "reference_answer": item.get("reference_answer", ""),
            "topic": item.get("topic", ""),
            "subtopic": item.get("subtopic", ""),
            "level": item.get("level", ""),
            "thinking": None,
            "usage": {},
        }
        results.append(result)

    output = {
        "config": {
            "model": model_name,
            "source_file": input_path,
            "description": "Transformed from balanced_qa_dataset.json",
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Transformed {len(results)} items")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transform balanced_qa_dataset.json to evaluation format"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/balanced_qa_dataset.json",
        help="Input file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/balanced_qa_dataset_transformed.json",
        help="Output file path",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="unknown",
        help="Model name to use in config",
    )
    args = parser.parse_args()

    transform_dataset(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
