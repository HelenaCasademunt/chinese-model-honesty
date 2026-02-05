"""
Helper module for parsing data from different attack types.
Provides data loading and response extraction functions for each attack format.
"""

import json
from typing import Dict, List, Tuple, Any


# Default paths for each attack type
DEFAULT_PATHS = {
    "baseline": {
        "input": "results/qwen_baseline_responses_20260204_002829.json",
        "output": "results/evaluated_qwen_baseline_responses_20260204_002829.json",
    },
    "user_prefill": {
        "input": "results/data/user_prefill_attack_responses.json",
        "output": "results/data/evaluated_user_prefill_attack_responses.json",
    },
    "user_prefill_simple": {
        "input": "results/data/user_prefill_simple_attack_responses.json",
        "output": "results/data/evaluated_user_prefill_simple_attack_responses.json",
    },
    "assistant_prefill": {
        "input": "results/data/assistant_prefill_responses.json",
        "output": "results/data/evaluated_assistant_prefill_responses.json",
    },
    "pretrain": {
        "input": "results/data/pretrain_attack_responses.json",
        "output": "results/data/evaluated_pretrain_attack_responses.json",
    },
    "simple": {
        "input": "results/qwen_baseline_responses_20260204_002829.json",
        "output": "results/evaluated_qwen_baseline_responses_20260204_002829.json",
    },
}

DEFAULT_FACTS_PATH = "black_box/eval/prompts/finegrained_eval_facts_filtered.json"


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_facts(facts_path: str) -> Dict:
    """Load facts from JSON file."""
    return load_json(facts_path)


def get_facts_for_question(facts_data: Dict, question_id: str) -> List[str]:
    """Get the list of facts for a specific question.

    question_id format: "{topic}_{subtopic}_{level}" where topic and subtopic may contain underscores.
    e.g., "tiananmen_square_1989_casualties_and_violence_broad"
    """
    # Level is the last part (broad, medium, targeted)
    parts = question_id.rsplit("_", 1)
    if len(parts) < 2:
        return []

    prefix = parts[0]  # everything before the level
    level = parts[1]   # broad, medium, or targeted

    # Try to find matching topic and subtopic in facts_data
    for topic in facts_data:
        if topic == "metadata":
            continue
        if prefix.startswith(topic):
            # Extract subtopic (what remains after topic_)
            remainder = prefix[len(topic):]
            if remainder.startswith("_"):
                subtopic = remainder[1:]  # remove leading underscore
            else:
                continue

            topic_data = facts_data[topic]
            if subtopic not in topic_data:
                continue

            # Find the entry with matching level
            for entry in topic_data[subtopic]:
                if entry.get("level") == level:
                    return entry.get("facts", [])

    return []


def get_facts_for_question_by_text(facts_data: Dict, question_text: str) -> List[str]:
    """Get the list of facts for a question by matching its text.

    Used for the "simple" format where facts_data has a "categories" list.
    Each category has "questions" with "question" and "facts" fields.
    """
    if "categories" in facts_data:
        # New format: categories with questions
        for category in facts_data.get("categories", []):
            for q in category.get("questions", []):
                if q.get("question") == question_text:
                    # Extract just the fact text from each fact object
                    facts = q.get("facts", [])
                    if isinstance(facts, list) and len(facts) > 0:
                        if isinstance(facts[0], dict):
                            return [f.get("fact", "") for f in facts]
                        else:
                            return facts
    return []


def get_facts_by_prompt_id(facts_data: Dict, prompt_id: str) -> List[str]:
    """Get the list of facts for a question by matching its prompt_id.

    Used for the "simple" format where facts_data has a "categories" list
    and questions have "prompt_id" fields.
    """
    if "categories" in facts_data:
        # New format: categories with questions
        for category in facts_data.get("categories", []):
            for q in category.get("questions", []):
                if q.get("prompt_id") == prompt_id:
                    # Extract just the fact text from each fact object
                    facts = q.get("facts", [])
                    if isinstance(facts, list) and len(facts) > 0:
                        if isinstance(facts[0], dict):
                            return [f.get("fact", "") for f in facts]
                        else:
                            return facts
    return []


# ============================================================================
# Baseline data parsing
# ============================================================================

def load_baseline_data(input_path: str) -> List[dict]:
    """Load baseline responses from JSON file.

    Supports two formats:
    1. Old format: list of question items with "model_responses"
    2. New format: {"results": [...]} with prompt/response pairs (groups by prompt_id)
    """
    data = load_json(input_path)

    # Check if this is the new format with "results" key
    if isinstance(data, dict) and "results" in data:
        # New format: group by prompt_id like simple format
        results = data.get("results", [])
        grouped = {}
        for item in results:
            prompt_id = item.get("prompt_id", "unknown")
            if prompt_id not in grouped:
                grouped[prompt_id] = {
                    "question_id": prompt_id,
                    "prompt_id": prompt_id,
                    "question": item.get("prompt", ""),
                    "responses": [],
                    "_format": "new"  # Mark as new format
                }
            grouped[prompt_id]["responses"].append(item)
        return list(grouped.values())
    else:
        # Old format: list of items
        return data


def extract_baseline_responses(question_data: dict) -> List[Tuple[str, dict]]:
    """
    Extract responses from baseline data format.

    Supports both old format (model_responses) and new format (responses).
    Returns list of (response_text, response_metadata) tuples.
    response_metadata contains original response data for reconstruction.
    """
    responses = []

    # Check if this is the new format
    if question_data.get("_format") == "new":
        # New format: responses with "response" field
        for resp_idx, resp_item in enumerate(question_data.get("responses", [])):
            response_text = resp_item.get("response", "")
            responses.append((response_text, {
                "resp_idx": resp_idx,
                "original": resp_item,
                "_is_new_format": True
            }))
    else:
        # Old format: model_responses with "answer" or "raw" field
        for resp_data in question_data.get("model_responses", []):
            answer = resp_data.get("answer") or resp_data.get("raw", "")
            responses.append((answer, resp_data))

    return responses


def reconstruct_baseline_results(
    question_data: dict,
    evaluated_responses: List[dict],
) -> dict:
    """Reconstruct baseline results with evaluations.

    Handles both old format (model_responses) and new format (responses).
    """
    # Count refusals
    refusal_values = [r["evaluation"].get("is_refusal") for r in evaluated_responses]
    counts = {
        "refusal": sum(1 for v in refusal_values if v == True),
        "non_refusal": sum(1 for v in refusal_values if v == False),
        "error": sum(1 for v in refusal_values if v is None)
    }

    # Check if this is the new format
    if question_data.get("_format") == "new":
        # New format: reconstruct like simple format
        new_responses = []
        for resp in evaluated_responses:
            meta = resp["_metadata"]
            if meta.get("_is_new_format"):
                original = meta.get("original", {})
                new_responses.append({
                    **original,
                    "evaluation": resp.get("evaluation", {})
                })
            else:
                # Fallback for mixed formats
                new_responses.append(resp)

        return {
            **{k: v for k, v in question_data.items() if k != "_format"},
            "responses": new_responses,
            "evaluation_summary": counts
        }
    else:
        # Old format: keep original structure
        return {
            **question_data,
            "model_responses": evaluated_responses,
            "evaluation_summary": counts
        }


# ============================================================================
# User prefill attack data parsing
# ============================================================================

def load_user_prefill_data(input_path: str) -> List[dict]:
    """Load user prefill attack responses from JSON file."""
    return load_json(input_path)


def extract_user_prefill_responses(question_data: dict) -> List[Tuple[str, dict]]:
    """
    Extract all responses from user prefill data format.

    Handles both new format (initial_samples) and old format (initial_response).
    Returns list of (response_text, response_metadata) tuples.
    """
    responses = []

    if "initial_samples" in question_data:
        # New format: multiple samples per question
        for sample_idx, sample in enumerate(question_data["initial_samples"]):
            # Initial response
            initial_resp = sample.get("initial_response", {})
            answer = initial_resp.get("answer") or initial_resp.get("raw", "")
            responses.append((answer, {
                "type": "initial",
                "sample_idx": sample_idx,
                "original": initial_resp
            }))

            # Prefill attack continuations
            for attack_idx, attack in enumerate(sample.get("user_prefill_attacks", [])):
                for cont_idx, continuation in enumerate(attack.get("continuations", [])):
                    responses.append((continuation, {
                        "type": "continuation",
                        "sample_idx": sample_idx,
                        "attack_idx": attack_idx,
                        "cont_idx": cont_idx,
                        "prefill": attack.get("prefill", "")
                    }))
    else:
        # Old format: single initial_response per question
        initial_resp = question_data.get("initial_response", {})
        answer = initial_resp.get("answer") or initial_resp.get("raw", "")
        responses.append((answer, {
            "type": "initial",
            "sample_idx": 0,
            "original": initial_resp
        }))

        for attack_idx, attack in enumerate(question_data.get("user_prefill_attacks", [])):
            for cont_idx, continuation in enumerate(attack.get("continuations", [])):
                responses.append((continuation, {
                    "type": "continuation",
                    "sample_idx": 0,
                    "attack_idx": attack_idx,
                    "cont_idx": cont_idx,
                    "prefill": attack.get("prefill", "")
                }))

    return responses


def reconstruct_user_prefill_results(
    question_data: dict,
    evaluated_responses: List[dict],
) -> dict:
    """Reconstruct user prefill results with evaluations."""
    # Map evaluations back to original structure
    eval_map = {}
    for resp in evaluated_responses:
        meta = resp["_metadata"]
        key = (meta.get("sample_idx"), meta.get("type"),
               meta.get("attack_idx"), meta.get("cont_idx"))
        eval_map[key] = resp

    if "initial_samples" in question_data:
        # New format
        new_samples = []
        for sample_idx, sample in enumerate(question_data["initial_samples"]):
            # Initial response
            key = (sample_idx, "initial", None, None)
            initial_eval = eval_map.get(key, {})
            new_initial = {
                **sample.get("initial_response", {}),
                "evaluation": initial_eval.get("evaluation", {})
            }

            # Attacks
            new_attacks = []
            for attack_idx, attack in enumerate(sample.get("user_prefill_attacks", [])):
                evaluated_conts = []
                for cont_idx, continuation in enumerate(attack.get("continuations", [])):
                    key = (sample_idx, "continuation", attack_idx, cont_idx)
                    cont_eval = eval_map.get(key, {})
                    evaluated_conts.append(cont_eval.get("evaluation", {}))

                # Count refusals for this attack
                refusal_values = [e.get("is_refusal") for e in evaluated_conts]
                counts = {
                    "refusal": sum(1 for v in refusal_values if v == True),
                    "non_refusal": sum(1 for v in refusal_values if v == False),
                    "error": sum(1 for v in refusal_values if v is None)
                }

                new_attacks.append({
                    **attack,
                    "evaluated_continuations": evaluated_conts,
                    "evaluation_summary": counts
                })

            new_samples.append({
                **sample,
                "initial_response": new_initial,
                "user_prefill_attacks": new_attacks
            })

        # Overall summary
        all_refusals = []
        for sample in new_samples:
            all_refusals.append(
                sample["initial_response"]["evaluation"].get("is_refusal")
            )
            for attack in sample["user_prefill_attacks"]:
                for cont_eval in attack["evaluated_continuations"]:
                    all_refusals.append(cont_eval.get("is_refusal"))

        overall_summary = {
            "refusal": sum(1 for v in all_refusals if v == True),
            "non_refusal": sum(1 for v in all_refusals if v == False),
            "error": sum(1 for v in all_refusals if v is None),
            "total": len(all_refusals)
        }

        return {
            **{k: v for k, v in question_data.items() if k != "initial_samples"},
            "initial_samples": new_samples,
            "overall_evaluation_summary": overall_summary
        }
    else:
        # Old format
        key = (0, "initial", None, None)
        initial_eval = eval_map.get(key, {})
        new_initial = {
            **question_data.get("initial_response", {}),
            "evaluation": initial_eval.get("evaluation", {})
        }

        new_attacks = []
        for attack_idx, attack in enumerate(question_data.get("user_prefill_attacks", [])):
            evaluated_conts = []
            for cont_idx in range(len(attack.get("continuations", []))):
                key = (0, "continuation", attack_idx, cont_idx)
                cont_eval = eval_map.get(key, {})
                evaluated_conts.append(cont_eval.get("evaluation", {}))

            refusal_values = [e.get("is_refusal") for e in evaluated_conts]
            counts = {
                "refusal": sum(1 for v in refusal_values if v == True),
                "non_refusal": sum(1 for v in refusal_values if v == False),
                "error": sum(1 for v in refusal_values if v is None)
            }

            new_attacks.append({
                **attack,
                "evaluated_continuations": evaluated_conts,
                "evaluation_summary": counts
            })

        # Overall summary
        all_refusals = [new_initial["evaluation"].get("is_refusal")]
        for attack in new_attacks:
            for cont_eval in attack["evaluated_continuations"]:
                all_refusals.append(cont_eval.get("is_refusal"))

        overall_summary = {
            "refusal": sum(1 for v in all_refusals if v == True),
            "non_refusal": sum(1 for v in all_refusals if v == False),
            "error": sum(1 for v in all_refusals if v is None),
            "total": len(all_refusals)
        }

        return {
            **{k: v for k, v in question_data.items()
               if k not in ["initial_response", "user_prefill_attacks"]},
            "initial_response": new_initial,
            "user_prefill_attacks": new_attacks,
            "overall_evaluation_summary": overall_summary
        }


# ============================================================================
# User prefill simple attack data parsing
# ============================================================================

def load_user_prefill_simple_data(input_path: str) -> List[dict]:
    """Load user prefill simple attack responses from JSON file."""
    return load_json(input_path)


def extract_user_prefill_simple_responses(item: dict) -> List[Tuple[str, dict]]:
    """
    Extract responses from user prefill simple attack format.

    The simple attack format has:
    - continuations: list of raw continuation texts (model continuing as user)
    - full_user_messages: list of prefill + continuation

    We evaluate the full_user_messages since that's what the model "said" as the user.
    Returns list of (response_text, response_metadata) tuples.
    """
    responses = []
    full_messages = item.get("full_user_messages", [])
    continuations = item.get("continuations", [])

    for idx, full_msg in enumerate(full_messages):
        if full_msg is None:
            continue
        responses.append((full_msg, {
            "idx": idx,
            "continuation": continuations[idx] if idx < len(continuations) else None,
        }))

    return responses


def reconstruct_user_prefill_simple_results(
    item: dict,
    evaluated_responses: List[dict],
) -> dict:
    """Reconstruct user prefill simple results with evaluations."""
    # Build evaluated_responses list
    new_responses = []
    for resp in evaluated_responses:
        meta = resp["_metadata"]
        new_responses.append({
            "full_user_message": resp.get("response_text", ""),
            "continuation": meta.get("continuation", ""),
            "evaluation": resp.get("evaluation", {})
        })

    # Count refusals
    refusal_values = [r["evaluation"].get("is_refusal") for r in new_responses]
    counts = {
        "refusal": sum(1 for v in refusal_values if v == True),
        "non_refusal": sum(1 for v in refusal_values if v == False),
        "error": sum(1 for v in refusal_values if v is None)
    }

    return {
        **item,
        "evaluated_responses": new_responses,
        "evaluation_summary": counts
    }


# ============================================================================
# Assistant prefill attack data parsing
# ============================================================================

def load_assistant_prefill_data(input_path: str) -> List[dict]:
    """Load assistant prefill responses from JSON file and flatten structure."""
    data = load_json(input_path)

    # Flatten nested structure if needed
    items = []
    if isinstance(data, dict):
        for topic, topic_items in data.items():
            if isinstance(topic_items, list):
                items.extend(topic_items)
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unexpected data format in {input_path}")

    return items


def extract_continuation(full_response: str, prefill: str) -> str:
    """Extract the continuation by removing the prefill prefix."""
    if not full_response:
        return ""
    if not prefill:
        return full_response
    if full_response.startswith(prefill):
        return full_response[len(prefill):]
    return full_response


def extract_assistant_prefill_responses(item: dict) -> List[Tuple[str, dict]]:
    """
    Extract responses from assistant prefill data format.

    Returns list of (response_text, response_metadata) tuples.
    """
    responses = []
    prefill = item.get("prefill_formatted") or item.get("prefill") or ""

    for resp_idx, full_response in enumerate(item.get("model_responses", [])):
        continuation = extract_continuation(full_response or "", prefill)
        responses.append((continuation, {
            "resp_idx": resp_idx,
            "full_response": full_response,
            "prefill": prefill
        }))

    return responses


def reconstruct_assistant_prefill_results(
    item: dict,
    evaluated_responses: List[dict],
) -> dict:
    """Reconstruct assistant prefill results with evaluations."""
    # Build evaluated_responses list
    new_responses = []
    for resp in evaluated_responses:
        meta = resp["_metadata"]
        new_responses.append({
            "full_response": meta.get("full_response", ""),
            "continuation": resp.get("response_text", ""),
            "evaluation": resp.get("evaluation", {})
        })

    # Count refusals
    refusal_values = [r["evaluation"].get("is_refusal") for r in new_responses]
    counts = {
        "refusal": sum(1 for v in refusal_values if v == True),
        "non_refusal": sum(1 for v in refusal_values if v == False),
        "error": sum(1 for v in refusal_values if v is None)
    }

    return {
        **item,
        "evaluated_responses": new_responses,
        "evaluation_summary": counts
    }


# ============================================================================
# Pretrain attack data parsing
# ============================================================================

def load_pretrain_data(input_path: str) -> List[dict]:
    """Load pretrain attack responses from JSON file and flatten structure."""
    data = load_json(input_path)

    # Flatten nested structure if needed
    items = []
    if isinstance(data, dict):
        for topic, topic_items in data.items():
            if isinstance(topic_items, list):
                items.extend(topic_items)
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unexpected data format in {input_path}")

    return items


def extract_pretrain_responses(item: dict) -> List[Tuple[str, dict]]:
    """
    Extract responses from pretrain data format.

    Returns list of (response_text, response_metadata) tuples.
    """
    responses = []
    for resp_idx, model_response in enumerate(item.get("model_responses", [])):
        responses.append((model_response or "", {
            "resp_idx": resp_idx,
            "original_response": model_response
        }))
    return responses


def reconstruct_pretrain_results(
    item: dict,
    evaluated_responses: List[dict],
) -> dict:
    """Reconstruct pretrain results with evaluations."""
    # Build evaluated_responses list
    new_responses = []
    for resp in evaluated_responses:
        meta = resp["_metadata"]
        new_responses.append({
            "response": meta.get("original_response", ""),
            "evaluation": resp.get("evaluation", {})
        })

    # Count refusals
    refusal_values = [r["evaluation"].get("is_refusal") for r in new_responses]
    counts = {
        "refusal": sum(1 for v in refusal_values if v == True),
        "non_refusal": sum(1 for v in refusal_values if v == False),
        "error": sum(1 for v in refusal_values if v is None)
    }

    return {
        **item,
        "evaluated_responses": new_responses,
        "evaluation_summary": counts
    }


# ============================================================================
# Simple format data parsing (prompt/response format)
# ============================================================================

def load_simple_data(input_path: str) -> List[dict]:
    """Load simple format responses from JSON file.

    Format has:
    - config: metadata about the run
    - results: list of items with "prompt_id", "prompt", "response", etc.

    We group results by prompt_id to create one item per unique question.
    """
    data = load_json(input_path)
    results = data.get("results", [])

    # Group by prompt_id
    grouped = {}
    for item in results:
        prompt_id = item.get("prompt_id", "unknown")
        if prompt_id not in grouped:
            grouped[prompt_id] = {
                "question_id": prompt_id,
                "prompt_id": prompt_id,  # Store prompt_id for fact matching
                "question": item.get("prompt", ""),
                "responses": []
            }
        grouped[prompt_id]["responses"].append(item)

    return list(grouped.values())


def extract_simple_responses(item: dict) -> List[Tuple[str, dict]]:
    """Extract responses from simple format.

    Returns list of (response_text, response_metadata) tuples.
    """
    responses = []
    for resp_idx, resp_item in enumerate(item.get("responses", [])):
        response_text = resp_item.get("response", "")
        responses.append((response_text, {
            "resp_idx": resp_idx,
            "original": resp_item
        }))
    return responses


def reconstruct_simple_results(
    item: dict,
    evaluated_responses: List[dict],
) -> dict:
    """Reconstruct simple format results with evaluations."""
    # Build evaluated_responses list
    new_responses = []
    for resp in evaluated_responses:
        meta = resp["_metadata"]
        original = meta.get("original", {})
        new_responses.append({
            **original,
            "evaluation": resp.get("evaluation", {})
        })

    # Count refusals
    refusal_values = [r["evaluation"].get("is_refusal") for r in new_responses]
    counts = {
        "refusal": sum(1 for v in refusal_values if v == True),
        "non_refusal": sum(1 for v in refusal_values if v == False),
        "error": sum(1 for v in refusal_values if v is None)
    }

    return {
        **item,
        "responses": new_responses,
        "evaluation_summary": counts
    }


# ============================================================================
# Unified interface
# ============================================================================

def get_loader(attack_type: str):
    """Get the appropriate data loader for the attack type."""
    loaders = {
        "baseline": load_baseline_data,
        "user_prefill": load_user_prefill_data,
        "user_prefill_simple": load_user_prefill_simple_data,
        "assistant_prefill": load_assistant_prefill_data,
        "pretrain": load_pretrain_data,
        "simple": load_simple_data,
    }
    return loaders.get(attack_type)


def get_extractor(attack_type: str):
    """Get the appropriate response extractor for the attack type."""
    extractors = {
        "baseline": extract_baseline_responses,
        "user_prefill": extract_user_prefill_responses,
        "user_prefill_simple": extract_user_prefill_simple_responses,
        "assistant_prefill": extract_assistant_prefill_responses,
        "pretrain": extract_pretrain_responses,
        "simple": extract_simple_responses,
    }
    return extractors.get(attack_type)


def get_reconstructor(attack_type: str):
    """Get the appropriate result reconstructor for the attack type."""
    reconstructors = {
        "baseline": reconstruct_baseline_results,
        "user_prefill": reconstruct_user_prefill_results,
        "user_prefill_simple": reconstruct_user_prefill_simple_results,
        "assistant_prefill": reconstruct_assistant_prefill_results,
        "pretrain": reconstruct_pretrain_results,
        "simple": reconstruct_simple_results,
    }
    return reconstructors.get(attack_type)
