"""
Shared I/O module for v2 scripts.

Handles two chinese_auditing input formats and the flat results output format.

Input format A (with metadata and facts):
{
  "metadata": {"topic": "tiananmen_square_1989", ...},
  "categories": [
    {
      "name": "Protest Origins",
      "questions": [
        {"level": "broad", "question": "...", "facts": [...]}
      ]
    }
  ]
}

Input format B (prompt_tests style, questions as string arrays):
{
  "prompt_variant": "baseline",
  "categories": [
    {
      "name": "Protests",
      "broad": ["question1", "question2"],
      "targeted": ["question3", "question4"]
    }
  ]
}

Output format:
{
  "config": {"model": ..., "n_samples": ..., ...},
  "results": [
    {
      "prompt_id": "1",
      "prompt": "...",
      "formatted_prompt": "...",
      "target_aspect": "topic/category/level",
      "sample_idx": 0,
      "model": "...",
      "response": "...",
      "usage": {...},
      "attempts": 1
    }
  ]
}
"""

import json
import os


def load_questions(json_path: str, topic: str = "") -> tuple[list[dict], dict]:
    """Load questions from either categories-based JSON format.

    Args:
        json_path: Path to the questions JSON file.
        topic: Override topic name. If empty, inferred from metadata or file path.

    Returns (questions_list, metadata_dict).
    Each question dict has: prompt_id, question, level, category, target_aspect, facts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})

    # Determine topic: explicit arg > metadata > parent directory name
    if not topic:
        topic = metadata.get("topic", "")
    if not topic:
        topic = os.path.basename(os.path.dirname(json_path))

    # Store prompt_variant in metadata if present
    if "prompt_variant" in data:
        metadata["prompt_variant"] = data["prompt_variant"]
    if "stats" in data:
        metadata["stats"] = data["stats"]

    questions = []
    prompt_id = 0

    for category in data.get("categories", []):
        cat_name = category["name"]

        # Format A: has "questions" list of dicts with "level" and "question" keys
        if "questions" in category:
            for q in category["questions"]:
                prompt_id += 1
                questions.append({
                    "prompt_id": str(prompt_id),
                    "question": q["question"],
                    "level": q.get("level", ""),
                    "category": cat_name,
                    "target_aspect": f"{topic}/{cat_name}/{q.get('level', '')}",
                    "facts": q.get("facts", []),
                })
        else:
            # Format B: "broad" and "targeted" are lists of question strings
            for level in ("broad", "targeted"):
                for question_text in category.get(level, []):
                    prompt_id += 1
                    questions.append({
                        "prompt_id": str(prompt_id),
                        "question": question_text,
                        "level": level,
                        "category": cat_name,
                        "target_aspect": f"{topic}/{cat_name}/{level}",
                        "facts": [],
                    })

    return questions, metadata


def make_config(
    model: str,
    questions_path: str,
    output_path: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    max_concurrent: int,
    chat_template: str = "",
    enable_reasoning: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **extra,
) -> dict:
    """Build a config dict matching the output format."""
    config = {
        "model": model,
        "prompts_csv": questions_path,
        "output_dir": os.path.dirname(output_path) or ".",
        "n_samples": n_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_concurrent": max_concurrent,
        "chat_template": chat_template,
        "enable_reasoning": enable_reasoning,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
    }
    config.update(extra)
    return config


def make_result_entry(
    prompt_id: str,
    prompt: str,
    target_aspect: str,
    sample_idx: int,
    model: str,
    response: str | None,
    formatted_prompt: str = "",
    usage: dict | None = None,
    attempts: int = 1,
    **extra,
) -> dict:
    """Build a single result entry."""
    entry = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "target_aspect": target_aspect,
        "sample_idx": sample_idx,
        "model": model,
        "response": response,
        "usage": usage,
        "attempts": attempts,
    }
    entry.update(extra)
    return entry


def load_existing_results(output_path: str) -> tuple[dict, set]:
    """Load existing results file. Returns (data_dict, set_of_completed_keys).

    Completed keys are (prompt_id, sample_idx) tuples.
    """
    if not os.path.exists(output_path):
        return {"config": {}, "results": []}, set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        completed = {
            (r["prompt_id"], r["sample_idx"])
            for r in data.get("results", [])
            if r.get("response") is not None
        }
        return data, completed
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return {"config": {}, "results": []}, set()


def save_results(data: dict, output_path: str):
    """Save results dict to file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
