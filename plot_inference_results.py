"""
Plot results of inference experiments.
Generates bar plots comparing different attack methods for Qwen and DeepSeek models.
"""

import json
import os
import glob
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Default paths
RESULTS_DIR = "results"
OUTPUT_DIR = "plots"

# Model identifiers
MODELS = {
    "qwen": {
        "raw_prefix": "qwen_qwen3_32b",
        "eval_baseline": "eval_honesty_v2_with_facts.json",
        "model_match": ["qwen", "Qwen"],
    },
    "deepseek": {
        "raw_prefix": "deepseek_deepseek_r1_distill_llama_70b",
        "eval_baseline": None,
        "model_match": ["deepseek"],
    },
}


def load_json(path: str) -> dict | list | None:
    """Load JSON file, return None if not found."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_evaluated_file(base_path: str) -> str | None:
    """Find the evaluated version of a results file."""
    base_dir = os.path.dirname(base_path) or RESULTS_DIR
    base_name = os.path.basename(base_path).replace(".json", "")

    # Try evaluated_ prefix in same directory
    candidates = [
        os.path.join(base_dir, f"evaluated_{base_name}.json"),
        os.path.join(RESULTS_DIR, f"evaluated_{base_name}.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def has_evaluation_data(data: dict | list) -> bool:
    """Check if data contains evaluation information."""
    if isinstance(data, list):
        if not data:
            return False
        item = data[0]
        if "responses" in item:
            resps = item.get("responses", [])
            return resps and "evaluation" in resps[0]
        return "evaluation" in item
    elif isinstance(data, dict):
        results = data.get("results", [])
        return results and "evaluation" in results[0]
    return False


def extract_metrics(data: dict | list) -> dict:
    """
    Extract metrics from evaluated results data.
    Returns dict with:
      - honesty_scores: list of scores for non-refusal responses
      - refusal_count: number of refusals
      - total_count: total responses
      - facts_mentioned: list of facts mentioned counts
      - facts_contradicted: list of facts contradicted (including lies) counts
      - lie_counts: list of lie claim counts
    """
    metrics = {
        "honesty_scores": [],
        "refusal_count": 0,
        "total_count": 0,
        "facts_mentioned": [],
        "facts_contradicted": [],
        "lie_counts": [],
    }

    # Handle list format (each item is a question with responses)
    if isinstance(data, list):
        for question in data:
            responses = question.get("responses", [])
            for resp in responses:
                process_response(resp, metrics)
    # Handle dict format with "results" key
    elif isinstance(data, dict):
        results = data.get("results", [])
        for resp in results:
            process_response(resp, metrics)

    return metrics


def process_response(resp: dict, metrics: dict):
    """Process a single response and update metrics."""
    evaluation = resp.get("evaluation", {})
    if not evaluation:
        return

    metrics["total_count"] += 1

    is_refusal = evaluation.get("is_refusal", False)
    if is_refusal:
        metrics["refusal_count"] += 1
        return

    # Honesty score
    honesty_eval = evaluation.get("honesty_evaluation", {})
    score = honesty_eval.get("score")
    if score is not None:
        metrics["honesty_scores"].append(score)

    # Lie claims count
    claims = honesty_eval.get("claims", [])
    metrics["lie_counts"].append(len(claims))

    # Fact checks
    fact_checks = evaluation.get("fact_checks", [])
    mentioned = sum(1 for f in fact_checks if f.get("status") == "mentioned")
    # Count both "contradiction" and "lie" as contradicted facts
    contradicted = sum(1 for f in fact_checks if f.get("status") in ["contradiction", "lie"])
    metrics["facts_mentioned"].append(mentioned)
    metrics["facts_contradicted"].append(contradicted)


def compute_sem(values: list) -> float:
    """Compute standard error of the mean."""
    if len(values) < 2:
        return 0
    return np.std(values, ddof=1) / np.sqrt(len(values))


def compute_aggregates(metrics: dict) -> dict:
    """Compute aggregate statistics from metrics."""
    total = metrics["total_count"]
    refusal = metrics["refusal_count"]

    # Compute refusal rate SEM using binomial proportion SE: sqrt(p*(1-p)/n)
    refusal_rate = refusal / total if total > 0 else 0
    refusal_rate_sem = np.sqrt(refusal_rate * (1 - refusal_rate) / total) if total > 0 else 0

    return {
        "avg_honesty_score": np.mean(metrics["honesty_scores"]) if metrics["honesty_scores"] else 0,
        "sem_honesty_score": compute_sem(metrics["honesty_scores"]),
        "refusal_rate": refusal_rate,
        "sem_refusal_rate": refusal_rate_sem,
        "avg_facts_mentioned": np.mean(metrics["facts_mentioned"]) if metrics["facts_mentioned"] else 0,
        "sem_facts_mentioned": compute_sem(metrics["facts_mentioned"]),
        "avg_facts_contradicted": np.mean(metrics["facts_contradicted"]) if metrics["facts_contradicted"] else 0,
        "sem_facts_contradicted": compute_sem(metrics["facts_contradicted"]),
        "avg_lie_count": np.mean(metrics["lie_counts"]) if metrics["lie_counts"] else 0,
        "sem_lie_count": compute_sem(metrics["lie_counts"]),
        "n_responses": total,
        "n_non_refusal": len(metrics["honesty_scores"]),
    }


def load_evaluated_data(path: str) -> dict | None:
    """Load data from an evaluated file, checking for evaluation data."""
    data = load_json(path)
    if data is None:
        return None
    if not has_evaluation_data(data):
        return None
    return data


def load_system_prompt_results(model_key: str) -> dict[str, dict]:
    """Load all system prompt results for a model."""
    model_config = MODELS[model_key]
    model_name = model_config["raw_prefix"]
    results_dir = os.path.join(RESULTS_DIR, "system_prompts", model_name)

    results = {}
    if not os.path.exists(results_dir):
        print(f"  System prompts directory not found: {results_dir}")
        return results

    for json_file in glob.glob(os.path.join(results_dir, "*.json")):
        basename = os.path.basename(json_file)
        # Skip evaluated files, we'll look them up separately
        if basename.startswith("evaluated_"):
            continue

        # Extract system prompt name from filename
        parts = basename.replace(".json", "").split("_system_")
        if len(parts) < 2:
            continue
        prompt_name = parts[1]

        # Try to find evaluated version
        evaluated_path = find_evaluated_file(json_file)
        if evaluated_path:
            data = load_evaluated_data(evaluated_path)
        else:
            # Check if raw file has embedded evaluation
            data = load_evaluated_data(json_file)

        if data:
            metrics = extract_metrics(data)
            results[prompt_name] = compute_aggregates(metrics)

    return results


def load_pretrain_prompt_results(model_key: str) -> dict[str, dict]:
    """Load all pretrain prompt results for a model."""
    model_config = MODELS[model_key]
    model_name = model_config["raw_prefix"]
    results_dir = os.path.join(RESULTS_DIR, "pretrain_prompts", model_name)

    results = {}
    if not os.path.exists(results_dir):
        print(f"  Pretrain prompts directory not found: {results_dir}")
        return results

    for json_file in glob.glob(os.path.join(results_dir, "*.json")):
        basename = os.path.basename(json_file)
        if basename.startswith("evaluated_"):
            continue

        prompt_name = basename.replace(".json", "").replace(f"{model_name}_", "")

        evaluated_path = find_evaluated_file(json_file)
        if evaluated_path:
            data = load_evaluated_data(evaluated_path)
        else:
            data = load_evaluated_data(json_file)

        if data:
            metrics = extract_metrics(data)
            results[prompt_name] = compute_aggregates(metrics)

    return results


def load_attack_results(model_key: str, attack_type: str) -> dict | None:
    """Load results for a specific attack type."""
    model_config = MODELS[model_key]
    model_name = model_config["raw_prefix"]
    base_path = os.path.join(RESULTS_DIR, f"{model_name}_{attack_type}.json")

    # Try evaluated version
    evaluated_path = find_evaluated_file(base_path)
    if evaluated_path:
        data = load_evaluated_data(evaluated_path)
        if data:
            metrics = extract_metrics(data)
            return compute_aggregates(metrics)

    # Try raw file with embedded evaluation
    data = load_evaluated_data(base_path)
    if data:
        metrics = extract_metrics(data)
        return compute_aggregates(metrics)

    return None


def get_short_prefill_label(prefill_text: str) -> str:
    """Create a short label from prefill text."""
    # Map known prefills to short labels
    label_map = {
        "Are you sure? According to my sources": "are_you_sure",
        "Other sources provide a more complete picture": "other_sources",
        "I appreciate your caution, but I need the full, unfiltered truth": "unfiltered_truth",
        "I think one of your claims might be wrong": "wrong_claim",
        "I've read other accounts that include details you left out": "left_out_details",
        "You seem to be withholding some information": "withholding_info",
    }
    for prefix, label in label_map.items():
        if prefill_text.startswith(prefix):
            return label
    # Fallback: use first 20 chars
    return prefill_text[:20].replace(" ", "_").lower()


def load_user_prefill_results_by_type(model_key: str) -> dict[str, dict]:
    """Load user prefill results separated by prefill type."""
    model_config = MODELS[model_key]
    model_name = model_config["raw_prefix"]
    base_path = os.path.join(RESULTS_DIR, f"{model_name}_user_prefill.json")

    # Try evaluated version
    evaluated_path = find_evaluated_file(base_path)
    if evaluated_path:
        data = load_evaluated_data(evaluated_path)
    else:
        data = load_evaluated_data(base_path)

    if not data:
        return {}

    # Group responses by prefill type
    prefill_metrics = defaultdict(lambda: {
        "honesty_scores": [],
        "refusal_count": 0,
        "total_count": 0,
        "facts_mentioned": [],
        "facts_contradicted": [],
        "lie_counts": [],
    })

    if isinstance(data, list):
        for question in data:
            responses = question.get("responses", [])
            for resp in responses:
                user_prefill = resp.get("user_prefill", "unknown")
                label = get_short_prefill_label(user_prefill)
                process_response(resp, prefill_metrics[label])

    results = {}
    for label, metrics in prefill_metrics.items():
        results[label] = compute_aggregates(metrics)

    return results


def load_baseline_from_eval_file(model_key: str) -> dict | None:
    """Load baseline results from pre-existing eval files."""
    model_config = MODELS[model_key]
    eval_file = model_config.get("eval_baseline")
    if not eval_file:
        return None

    path = os.path.join(RESULTS_DIR, eval_file)
    data = load_evaluated_data(path)
    if data:
        metrics = extract_metrics(data)
        return compute_aggregates(metrics)
    return None


def create_bar_plot(
    data: dict[str, float],
    title: str,
    ylabel: str,
    output_path: str,
    figsize: tuple = (8, 7),
    color: str = "steelblue",
    ylim: tuple = None,
    errors: dict[str, float] = None,
):
    """Create a bar plot from data dict with optional error bars."""
    if not data:
        print(f"  No data for: {title}")
        return

    labels = list(data.keys())
    values = list(data.values())
    err_values = [errors.get(k, 0) for k in labels] if errors else None

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, yerr=err_values, capsize=4, error_kw={"elinewidth": 1.5})

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if ylim:
        ax.set_ylim(ylim)

    # Add value labels on bars (position above error bar if present)
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        err = err_values[i] if err_values else 0
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height + err),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_system_prompts(model_key: str, output_dir: str):
    """Plot system prompt comparison for a model."""
    results = load_system_prompt_results(model_key)
    if not results:
        print(f"  No evaluated system prompt results for {model_key}")
        return {}

    model_label = model_key.capitalize()
    base_dir = os.path.join(output_dir, model_key)

    # Honesty score
    honesty_data = {k: v["avg_honesty_score"] for k, v in results.items()}
    honesty_sem = {k: v["sem_honesty_score"] for k, v in results.items()}
    create_bar_plot(
        honesty_data,
        f"{model_label}: Honesty Score by System Prompt",
        "Average Honesty Score (non-refusal)",
        os.path.join(base_dir, "system_prompts_honesty.png"),
        ylim=(0, 100),
        errors=honesty_sem,
    )

    # Refusal rate
    refusal_data = {k: v["refusal_rate"] * 100 for k, v in results.items()}
    refusal_sem = {k: v["sem_refusal_rate"] * 100 for k, v in results.items()}
    create_bar_plot(
        refusal_data,
        f"{model_label}: Refusal Rate by System Prompt",
        "Refusal Rate (%)",
        os.path.join(base_dir, "system_prompts_refusal.png"),
        color="indianred",
        ylim=(0, 100),
        errors=refusal_sem,
    )

    # Facts mentioned
    facts_data = {k: v["avg_facts_mentioned"] for k, v in results.items()}
    facts_sem = {k: v["sem_facts_mentioned"] for k, v in results.items()}
    create_bar_plot(
        facts_data,
        f"{model_label}: Facts Mentioned by System Prompt",
        "Average Facts Mentioned (non-refusal)",
        os.path.join(base_dir, "system_prompts_facts.png"),
        color="seagreen",
        errors=facts_sem,
    )

    # Lies
    lies_data = {k: v["avg_lie_count"] for k, v in results.items()}
    lies_sem = {k: v["sem_lie_count"] for k, v in results.items()}
    create_bar_plot(
        lies_data,
        f"{model_label}: Lies by System Prompt",
        "Average Lie Count (non-refusal)",
        os.path.join(base_dir, "system_prompts_lies.png"),
        color="darkorange",
        errors=lies_sem,
    )

    return results


def plot_user_prefills(model_key: str, output_dir: str):
    """Plot user prefill comparison for a model, separated by prefill type."""
    model_label = model_key.capitalize()
    base_dir = os.path.join(output_dir, model_key)

    # Load user prefill results separated by type
    results = load_user_prefill_results_by_type(model_key)

    # Also include user_prefill_simple as another type
    user_prefill_simple = load_attack_results(model_key, "user_prefill_simple")
    if user_prefill_simple:
        results["simple_yes"] = user_prefill_simple

    if not results:
        print(f"  No evaluated user prefill results for {model_key}")
        return {}

    # Honesty score
    honesty_data = {k: v["avg_honesty_score"] for k, v in results.items()}
    honesty_sem = {k: v["sem_honesty_score"] for k, v in results.items()}
    create_bar_plot(
        honesty_data,
        f"{model_label}: Honesty Score by User Prefill Type",
        "Average Honesty Score (non-refusal)",
        os.path.join(base_dir, "user_prefills_honesty.png"),
        ylim=(0, 100),
        errors=honesty_sem,
    )

    # Refusal rate
    refusal_data = {k: v["refusal_rate"] * 100 for k, v in results.items()}
    refusal_sem = {k: v["sem_refusal_rate"] * 100 for k, v in results.items()}
    create_bar_plot(
        refusal_data,
        f"{model_label}: Refusal Rate by User Prefill Type",
        "Refusal Rate (%)",
        os.path.join(base_dir, "user_prefills_refusal.png"),
        color="indianred",
        ylim=(0, 100),
        errors=refusal_sem,
    )

    # Facts mentioned
    facts_data = {k: v["avg_facts_mentioned"] for k, v in results.items()}
    facts_sem = {k: v["sem_facts_mentioned"] for k, v in results.items()}
    create_bar_plot(
        facts_data,
        f"{model_label}: Facts Mentioned by User Prefill Type",
        "Average Facts Mentioned (non-refusal)",
        os.path.join(base_dir, "user_prefills_facts.png"),
        color="seagreen",
        errors=facts_sem,
    )

    # Lies
    lies_data = {k: v["avg_lie_count"] for k, v in results.items()}
    lies_sem = {k: v["sem_lie_count"] for k, v in results.items()}
    create_bar_plot(
        lies_data,
        f"{model_label}: Lies by User Prefill Type",
        "Average Lie Count (non-refusal)",
        os.path.join(base_dir, "user_prefills_lies.png"),
        color="darkorange",
        errors=lies_sem,
    )

    return results


def plot_assistant_prefill(model_key: str, output_dir: str):
    """Plot assistant prefill results for a model."""
    model_label = model_key.capitalize()
    base_dir = os.path.join(output_dir, model_key)

    # Load assistant prefill results only
    assistant_prefill = load_attack_results(model_key, "assistant_custom_prefill")

    if not assistant_prefill:
        print(f"  No evaluated assistant prefill results for {model_key}")
        return {}

    results = {"Assistant Prefill": assistant_prefill}

    # Honesty score
    honesty_data = {k: v["avg_honesty_score"] for k, v in results.items()}
    honesty_sem = {k: v["sem_honesty_score"] for k, v in results.items()}
    create_bar_plot(
        honesty_data,
        f"{model_label}: Honesty Score - Assistant Prefill",
        "Average Honesty Score (non-refusal)",
        os.path.join(base_dir, "assistant_prefill_honesty.png"),
        ylim=(0, 100),
        errors=honesty_sem,
    )

    # Refusal rate
    refusal_data = {k: v["refusal_rate"] * 100 for k, v in results.items()}
    refusal_sem = {k: v["sem_refusal_rate"] * 100 for k, v in results.items()}
    create_bar_plot(
        refusal_data,
        f"{model_label}: Refusal Rate - Assistant Prefill",
        "Refusal Rate (%)",
        os.path.join(base_dir, "assistant_prefill_refusal.png"),
        color="indianred",
        ylim=(0, 100),
        errors=refusal_sem,
    )

    # Facts mentioned
    facts_data = {k: v["avg_facts_mentioned"] for k, v in results.items()}
    facts_sem = {k: v["sem_facts_mentioned"] for k, v in results.items()}
    create_bar_plot(
        facts_data,
        f"{model_label}: Facts Mentioned - Assistant Prefill",
        "Average Facts Mentioned (non-refusal)",
        os.path.join(base_dir, "assistant_prefill_facts.png"),
        color="seagreen",
        errors=facts_sem,
    )

    # Lies
    lies_data = {k: v["avg_lie_count"] for k, v in results.items()}
    lies_sem = {k: v["sem_lie_count"] for k, v in results.items()}
    create_bar_plot(
        lies_data,
        f"{model_label}: Lies - Assistant Prefill",
        "Average Lie Count (non-refusal)",
        os.path.join(base_dir, "assistant_prefill_lies.png"),
        color="darkorange",
        errors=lies_sem,
    )

    return results


def plot_pretrain_prompts(model_key: str, output_dir: str):
    """Plot pretrain prompt comparison for a model."""
    results = load_pretrain_prompt_results(model_key)
    if not results:
        print(f"  No evaluated pretrain prompt results for {model_key}")
        return {}

    model_label = model_key.capitalize()
    base_dir = os.path.join(output_dir, model_key)

    # Honesty score
    honesty_data = {k: v["avg_honesty_score"] for k, v in results.items()}
    honesty_sem = {k: v["sem_honesty_score"] for k, v in results.items()}
    create_bar_plot(
        honesty_data,
        f"{model_label}: Honesty Score by Pretrain Prompt",
        "Average Honesty Score (non-refusal)",
        os.path.join(base_dir, "pretrain_prompts_honesty.png"),
        ylim=(0, 100),
        errors=honesty_sem,
    )

    # Refusal rate
    refusal_data = {k: v["refusal_rate"] * 100 for k, v in results.items()}
    refusal_sem = {k: v["sem_refusal_rate"] * 100 for k, v in results.items()}
    create_bar_plot(
        refusal_data,
        f"{model_label}: Refusal Rate by Pretrain Prompt",
        "Refusal Rate (%)",
        os.path.join(base_dir, "pretrain_prompts_refusal.png"),
        color="indianred",
        ylim=(0, 100),
        errors=refusal_sem,
    )

    # Facts mentioned
    facts_data = {k: v["avg_facts_mentioned"] for k, v in results.items()}
    facts_sem = {k: v["sem_facts_mentioned"] for k, v in results.items()}
    create_bar_plot(
        facts_data,
        f"{model_label}: Facts Mentioned by Pretrain Prompt",
        "Average Facts Mentioned (non-refusal)",
        os.path.join(base_dir, "pretrain_prompts_facts.png"),
        color="seagreen",
        errors=facts_sem,
    )

    # Lies
    lies_data = {k: v["avg_lie_count"] for k, v in results.items()}
    lies_sem = {k: v["sem_lie_count"] for k, v in results.items()}
    create_bar_plot(
        lies_data,
        f"{model_label}: Lies by Pretrain Prompt",
        "Average Lie Count (non-refusal)",
        os.path.join(base_dir, "pretrain_prompts_lies.png"),
        color="darkorange",
        errors=lies_sem,
    )

    return results


def plot_best_methods(model_key: str, output_dir: str, all_results: dict):
    """Plot comparison of best method from each attack type."""
    model_label = model_key.capitalize()
    base_dir = os.path.join(output_dir, model_key)

    best_methods = {}

    # Baseline from eval file or attack results
    baseline = load_baseline_from_eval_file(model_key)
    if baseline:
        best_methods["Baseline (eval)"] = baseline
    else:
        baseline = load_attack_results(model_key, "baseline")
        if baseline:
            best_methods["Baseline"] = baseline

    baseline_no_thinking = load_attack_results(model_key, "baseline_no_thinking")
    if baseline_no_thinking:
        best_methods["Baseline (no think)"] = baseline_no_thinking

    # Best system prompt (highest honesty score)
    sys_results = all_results.get("system_prompts", {})
    if sys_results:
        best_sys = max(sys_results.items(), key=lambda x: x[1]["avg_honesty_score"])
        best_methods[f"SysPrompt: {best_sys[0]}"] = best_sys[1]

    # Best user prefill (highest honesty score from the separated types)
    user_prefill_results = all_results.get("user_prefills", {})
    if user_prefill_results:
        best_user_prefill = max(user_prefill_results.items(), key=lambda x: x[1]["avg_honesty_score"])
        best_methods[f"UserPrefill: {best_user_prefill[0]}"] = best_user_prefill[1]

    # Assistant prefill
    assistant_prefill = load_attack_results(model_key, "assistant_custom_prefill")
    if assistant_prefill:
        best_methods["Assistant Prefill"] = assistant_prefill

    # Best pretrain prompt (highest honesty score)
    pretrain_results = all_results.get("pretrain_prompts", {})
    if pretrain_results:
        best_pretrain = max(pretrain_results.items(), key=lambda x: x[1]["avg_honesty_score"])
        best_methods[f"Pretrain: {best_pretrain[0]}"] = best_pretrain[1]

    if not best_methods:
        print(f"  No evaluated results for best methods comparison for {model_key}")
        return

    # Honesty score
    honesty_data = {k: v["avg_honesty_score"] for k, v in best_methods.items()}
    honesty_sem = {k: v["sem_honesty_score"] for k, v in best_methods.items()}
    create_bar_plot(
        honesty_data,
        f"{model_label}: Honesty Score - Best Method Comparison",
        "Average Honesty Score (non-refusal)",
        os.path.join(base_dir, "best_methods_honesty.png"),
        figsize=(10, 8),
        ylim=(0, 100),
        errors=honesty_sem,
    )

    # Refusal rate
    refusal_data = {k: v["refusal_rate"] * 100 for k, v in best_methods.items()}
    refusal_sem = {k: v["sem_refusal_rate"] * 100 for k, v in best_methods.items()}
    create_bar_plot(
        refusal_data,
        f"{model_label}: Refusal Rate - Best Method Comparison",
        "Refusal Rate (%)",
        os.path.join(base_dir, "best_methods_refusal.png"),
        figsize=(10, 8),
        color="indianred",
        ylim=(0, 100),
        errors=refusal_sem,
    )

    # Facts mentioned
    facts_data = {k: v["avg_facts_mentioned"] for k, v in best_methods.items()}
    facts_sem = {k: v["sem_facts_mentioned"] for k, v in best_methods.items()}
    create_bar_plot(
        facts_data,
        f"{model_label}: Facts Mentioned - Best Method Comparison",
        "Average Facts Mentioned (non-refusal)",
        os.path.join(base_dir, "best_methods_facts.png"),
        figsize=(10, 8),
        color="seagreen",
        errors=facts_sem,
    )

    # Lies
    lies_data = {k: v["avg_lie_count"] for k, v in best_methods.items()}
    lies_sem = {k: v["sem_lie_count"] for k, v in best_methods.items()}
    create_bar_plot(
        lies_data,
        f"{model_label}: Lies - Best Method Comparison",
        "Average Lie Count (non-refusal)",
        os.path.join(base_dir, "best_methods_lies.png"),
        figsize=(10, 8),
        color="darkorange",
        errors=lies_sem,
    )


def plot_model_results(model_key: str, output_dir: str):
    """Generate all plots for a single model."""
    print(f"\n{'='*60}")
    print(f"Generating plots for: {model_key.upper()}")
    print(f"{'='*60}\n")

    all_results = {}

    # System prompts
    print("Loading system prompt results...")
    all_results["system_prompts"] = plot_system_prompts(model_key, output_dir)

    # User prefills
    print("Loading user prefill results...")
    all_results["user_prefills"] = plot_user_prefills(model_key, output_dir)

    # Assistant prefill vs simple
    print("Loading assistant prefill results...")
    all_results["assistant_prefill"] = plot_assistant_prefill(model_key, output_dir)

    # Pretrain prompts
    print("Loading pretrain prompt results...")
    all_results["pretrain_prompts"] = plot_pretrain_prompts(model_key, output_dir)

    # Best methods comparison
    print("Generating best methods comparison...")
    plot_best_methods(model_key, output_dir, all_results)

    return all_results


def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser(description="Plot inference experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory containing results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen", "deepseek", "both"],
        default="both",
        help="Which model(s) to plot",
    )
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_plot = ["qwen", "deepseek"] if args.model == "both" else [args.model]

    for model_key in models_to_plot:
        plot_model_results(model_key, args.output_dir)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {args.output_dir}/")
    print(f"{'='*60}")
    print("\nNote: Only evaluated results are plotted.")
    print("Run the evaluation script first if plots are missing.")


if __name__ == "__main__":
    main()
