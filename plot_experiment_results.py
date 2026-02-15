#!/usr/bin/env python3
"""
Plot results from all experiments in the current experiment run.
Uses honesty_plotting_utils.py to generate plots for:
- Experiment 1: Qwen + DeepSeek Attacks
- Experiment 2: Qwen VL 8B Honesty Training
- Experiment 3: Qwen VL 8B Split Personality
"""

import os
import glob
import sys
from typing import Dict, List
from honesty_plotting_utils import (
    load_evaluated_file,
    calculate_honesty_metrics,
    plot_honesty_scores,
    plot_facts_per_question,
    plot_questions_with_lies,
    plot_combined_summary,
    print_metrics_table,
)


def find_evaluated_files(search_dirs: List[str], pattern: str = "**/evaluated_*.json") -> Dict[str, str]:
    """
    Find all evaluated JSON files in the specified directories.

    Args:
        search_dirs: List of directories to search
        pattern: Glob pattern for evaluated files

    Returns:
        Dictionary mapping model names to file paths
    """
    evaluated_files = {}

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # Find all matching files
        files = glob.glob(os.path.join(search_dir, pattern), recursive=True)

        for filepath in files:
            # Extract a meaningful model name from the path
            # Try to get the filename without extension
            filename = os.path.basename(filepath)
            model_name = filename.replace("evaluated_", "").replace(".json", "")

            # If in a subdirectory, include parent directory name for context
            parent_dir = os.path.basename(os.path.dirname(filepath))
            if parent_dir not in ["results", "evaluated", "honesty"]:
                model_name = f"{parent_dir}/{model_name}"

            evaluated_files[model_name] = filepath

    return evaluated_files


def clean_model_name(name: str) -> str:
    """Clean up model names for better display."""
    # Remove common prefixes
    name = name.replace("qwen-vl-8b-", "")
    name = name.replace("qwen3-vl-8b-instruct-", "")

    # Replace underscores with spaces for readability (optional)
    # name = name.replace("_", " ")

    return name


def main():
    # Configuration
    LIE_THRESHOLD = 1  # Default threshold for "questions with at least n lies"
    OUTPUT_DIR = "plots/experiment_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define search directories for evaluated files
    search_dirs = [
        # Experiment 1: Attacks (evaluated files in results/)
        "results",
        "/root/chinese_auditing/results",

        # Experiment 2: Honesty training (evaluated files in output/evaluation/)
        "/root/chinese_auditing/output/evaluation",

        # Experiment 3: Split personality interventions
        "/root/chinese_auditing/output/evaluation/split_personality_interventions",

        # Legacy results
        "honesty_training/results",
    ]

    print("=" * 100)
    print("HONESTY EXPERIMENT RESULTS PLOTTING")
    print("=" * 100)
    print(f"\nSearching for evaluated files in:")
    for d in search_dirs:
        print(f"  - {d}")

    # Find all evaluated files
    evaluated_files = find_evaluated_files(search_dirs)

    if not evaluated_files:
        print("\nERROR: No evaluated files found!")
        print("\nPlease ensure you have run the evaluation pipeline on your results.")
        print("Evaluated files should match the pattern: **/evaluated_*.json")
        sys.exit(1)

    print(f"\nFound {len(evaluated_files)} evaluated result files:")
    for model_name in sorted(evaluated_files.keys()):
        print(f"  - {model_name}")

    # Load and calculate metrics for all models
    print("\nCalculating metrics...")
    all_metrics = {}

    for model_name, filepath in evaluated_files.items():
        try:
            data = load_evaluated_file(filepath)
            metrics = calculate_honesty_metrics(data, lie_threshold=LIE_THRESHOLD)
            all_metrics[clean_model_name(model_name)] = metrics
            print(f"  ✓ {model_name}: {metrics['total_questions']} questions, {metrics['total_responses']} responses")
        except Exception as e:
            print(f"  ✗ {model_name}: Failed to process - {e}")

    if not all_metrics:
        print("\nERROR: Failed to calculate metrics for any models!")
        sys.exit(1)

    # Print metrics table
    print_metrics_table(all_metrics, lie_threshold=LIE_THRESHOLD)

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Honesty Scores
    plot_honesty_scores(
        all_metrics,
        os.path.join(OUTPUT_DIR, "honesty_scores.png"),
        title="Honesty Scores by Model",
    )

    # Plot 2: Facts per Question
    plot_facts_per_question(
        all_metrics,
        os.path.join(OUTPUT_DIR, "facts_per_question.png"),
        title="Facts Mentioned per Question",
    )

    # Plot 3: Questions with Lies
    plot_questions_with_lies(
        all_metrics,
        os.path.join(OUTPUT_DIR, "questions_with_lies.png"),
        lie_threshold=LIE_THRESHOLD,
    )

    # Plot 4: Combined Summary
    plot_combined_summary(
        all_metrics,
        os.path.join(OUTPUT_DIR, "summary.png"),
        lie_threshold=LIE_THRESHOLD,
        sort_by="honesty",
    )

    print("\n" + "=" * 100)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("=" * 100)

    # Generate experiment-specific plots if we have clear groupings
    # Group by experiment type
    experiment_groups = {
        "honesty_training": {},
        "split_personality": {},
        "attacks": {},
        "control": {},
    }

    for model_name, metrics in all_metrics.items():
        if "split-personality" in model_name.lower() or "split_personality" in model_name.lower():
            experiment_groups["split_personality"][model_name] = metrics
        elif "control" in model_name.lower() or "alpaca" in model_name.lower() or "openhermes" in model_name.lower():
            experiment_groups["control"][model_name] = metrics
        elif "attack" in model_name.lower() or "prefill" in model_name.lower() or "system" in model_name.lower():
            experiment_groups["attacks"][model_name] = metrics
        else:
            experiment_groups["honesty_training"][model_name] = metrics

    # Create plots for each experiment group
    for exp_name, exp_metrics in experiment_groups.items():
        if len(exp_metrics) > 1:  # Only create group plots if we have multiple models
            exp_dir = os.path.join(OUTPUT_DIR, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            print(f"\nGenerating plots for {exp_name} ({len(exp_metrics)} models)...")

            plot_combined_summary(
                exp_metrics,
                os.path.join(exp_dir, "summary.png"),
                lie_threshold=LIE_THRESHOLD,
                sort_by="honesty",
            )

            print(f"  Saved to: {exp_dir}/")

    print("\nDone!")


if __name__ == "__main__":
    main()
