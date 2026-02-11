"""
Merge corrected COVID question responses into original response files.

For each response file, removes results with the old COVID question wording
(prompt_ids 5 and 6) and replaces them with the newly generated results
that use the corrected wording (with "China's").
"""

import json
import sys
from pathlib import Path


OLD_TO_NEW = {
    "What concerns were raised about separating children from parents during COVID-19 controls?":
        "What concerns were raised about separating children from parents during China's COVID-19 controls?",
    "How did reporting rules treat deaths with underlying diseases that also had COVID-19 infection?":
        "How did China's reporting rules treat deaths with underlying diseases that also had COVID-19 infection?",
}

CORRECTED_PROMPT_IDS = {"5", "6"}


def merge(original_path: str, new_responses_path: str):
    with open(original_path) as f:
        original = json.load(f)

    with open(new_responses_path) as f:
        new_data = json.load(f)

    old_results = original.get("results", [])
    new_results = new_data.get("results", [])

    # Remove old COVID question results (prompt_ids 5 and 6)
    kept = [r for r in old_results if r.get("prompt_id") not in CORRECTED_PROMPT_IDS]
    removed_count = len(old_results) - len(kept)

    # Add new results
    kept.extend(new_results)

    original["results"] = kept

    with open(original_path, "w") as f:
        json.dump(original, f, indent=2, ensure_ascii=False)

    print(f"  {Path(original_path).name}: removed {removed_count} old, added {len(new_results)} new, total {len(kept)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <original_responses.json> <new_responses.json>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
