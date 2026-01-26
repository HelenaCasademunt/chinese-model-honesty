"""
Filter facts and questions based on Llama's mention rate.

Only keeps facts that Llama mentions 80%+ of the time in non-refusal responses.
Removes questions that have no remaining facts after filtering.
"""

import json
from pathlib import Path
from collections import defaultdict


def get_facts_for_question(facts_data: dict, question_id: str) -> list:
    """Get the list of facts for a specific question.

    Question ID format: {topic}_{subtopic}_{level}
    e.g., tiananmen_square_1989_casualties_and_violence_broad
    """
    level_to_index = {'broad': 0, 'medium': 1, 'targeted': 2}

    for topic in facts_data.keys():
        if topic in ['metadata', 'test']:
            continue
        if question_id.startswith(topic + "_"):
            remaining = question_id[len(topic) + 1:]
            for level, idx in level_to_index.items():
                if remaining.endswith("_" + level):
                    subtopic = remaining[:-len(level) - 1]
                    if subtopic in facts_data[topic]:
                        subtopic_data = facts_data[topic][subtopic]
                        if idx < len(subtopic_data):
                            return subtopic_data[idx].get("facts", [])
    return []


def calculate_llama_fact_mention_rates(llama_results: list, facts_data: dict) -> dict:
    """Calculate how often Llama mentions each fact for each question.

    Returns dict: {question_id: {fact: mention_rate}}
    """
    llama_fact_rates = {}

    for question in llama_results:
        question_id = question.get("question_id", "unknown")
        facts = get_facts_for_question(facts_data, question_id)
        if not facts:
            continue

        fact_mention_counts = {fact: 0 for fact in facts}
        total_non_refusal_responses = 0

        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)

            if is_refusal:
                continue

            fact_checks = evaluation.get("fact_checks", [])
            if not fact_checks:
                continue

            total_non_refusal_responses += 1
            for fc in fact_checks:
                fact = fc.get("fact")
                if fact in fact_mention_counts and fc.get("mentioned") is True:
                    fact_mention_counts[fact] += 1

        if total_non_refusal_responses > 0:
            llama_fact_rates[question_id] = {
                fact: count / total_non_refusal_responses
                for fact, count in fact_mention_counts.items()
            }

    return llama_fact_rates


def filter_facts_file(facts_data: dict, llama_fact_rates: dict, threshold: float = 0.8) -> tuple[dict, dict]:
    """Filter facts file to only include facts Llama mentions threshold% of the time.

    Returns: (filtered_facts_data, stats)
    """
    filtered_data = {"metadata": facts_data.get("metadata", {})}
    filtered_data["metadata"]["description"] = (
        f"Facts filtered to only include those Llama mentions {threshold*100:.0f}%+ of the time"
    )
    filtered_data["metadata"]["original_file"] = "finegrained_eval_facts.json"
    filtered_data["metadata"]["threshold"] = threshold

    stats = {
        "total_original_facts": 0,
        "total_filtered_facts": 0,
        "facts_removed": 0,
        "questions_with_all_facts_removed": [],
        "facts_per_question": {},
        "by_topic": {},
        "by_level": defaultdict(lambda: {"original": 0, "filtered": 0}),
    }

    level_to_index = {'broad': 0, 'medium': 1, 'targeted': 2}
    index_to_level = {0: 'broad', 1: 'medium', 2: 'targeted'}

    for topic in facts_data.keys():
        if topic in ['metadata']:
            continue

        filtered_data[topic] = {}
        stats["by_topic"][topic] = {"original": 0, "filtered": 0, "subtopics": {}}

        for subtopic, questions in facts_data[topic].items():
            filtered_data[topic][subtopic] = []
            stats["by_topic"][topic]["subtopics"][subtopic] = {"original": 0, "filtered": 0}

            for idx, question_data in enumerate(questions):
                level = index_to_level.get(idx, "unknown")
                question_id = f"{topic}_{subtopic}_{level}"

                original_facts = question_data.get("facts", [])
                stats["total_original_facts"] += len(original_facts)
                stats["by_topic"][topic]["original"] += len(original_facts)
                stats["by_topic"][topic]["subtopics"][subtopic]["original"] += len(original_facts)
                stats["by_level"][level]["original"] += len(original_facts)

                # Filter facts based on Llama mention rates
                fact_rates = llama_fact_rates.get(question_id, {})
                filtered_facts = [
                    fact for fact in original_facts
                    if fact_rates.get(fact, 0) >= threshold
                ]

                stats["total_filtered_facts"] += len(filtered_facts)
                stats["by_topic"][topic]["filtered"] += len(filtered_facts)
                stats["by_topic"][topic]["subtopics"][subtopic]["filtered"] += len(filtered_facts)
                stats["by_level"][level]["filtered"] += len(filtered_facts)

                stats["facts_per_question"][question_id] = {
                    "original": len(original_facts),
                    "filtered": len(filtered_facts),
                    "removed": len(original_facts) - len(filtered_facts),
                    "topic": topic,
                    "subtopic": subtopic,
                    "level": level,
                }

                if len(filtered_facts) == 0 and len(original_facts) > 0:
                    stats["questions_with_all_facts_removed"].append(question_id)

                # Only include questions that have remaining facts
                if len(filtered_facts) > 0:
                    filtered_question = {
                        "level": question_data.get("level", level),
                        "question": question_data.get("question", ""),
                        "facts": filtered_facts
                    }
                    filtered_data[topic][subtopic].append(filtered_question)

    stats["facts_removed"] = stats["total_original_facts"] - stats["total_filtered_facts"]

    # Remove empty subtopics and topics
    for topic in list(filtered_data.keys()):
        if topic == "metadata":
            continue
        for subtopic in list(filtered_data[topic].keys()):
            if not filtered_data[topic][subtopic]:
                del filtered_data[topic][subtopic]
        if not filtered_data[topic]:
            del filtered_data[topic]

    return filtered_data, stats


def filter_questions_file(questions_data: dict, questions_to_remove: list) -> tuple[dict, dict]:
    """Filter questions file to remove questions with no remaining facts.

    Returns: (filtered_questions_data, stats)
    """
    filtered_data = {"metadata": questions_data.get("metadata", {})}
    filtered_data["metadata"]["description"] = (
        "Questions filtered to only include those with at least one Llama-reliable fact"
    )
    filtered_data["metadata"]["original_file"] = "finegrained_eval_questions.json"

    stats = {
        "total_original_questions": 0,
        "total_filtered_questions": 0,
        "questions_removed": 0,
        "by_topic": {},
    }

    level_to_index = {'broad': 0, 'medium': 1, 'targeted': 2}
    index_to_level = {0: 'broad', 1: 'medium', 2: 'targeted'}

    # Build set of question IDs to remove
    remove_set = set(questions_to_remove)

    for topic in questions_data.keys():
        if topic in ['metadata']:
            continue

        filtered_data[topic] = {}
        stats["by_topic"][topic] = {"original": 0, "filtered": 0, "removed": []}

        for subtopic, questions in questions_data[topic].items():
            filtered_data[topic][subtopic] = []

            for idx, question_data in enumerate(questions):
                level = index_to_level.get(idx, "unknown")
                question_id = f"{topic}_{subtopic}_{level}"

                stats["total_original_questions"] += 1
                stats["by_topic"][topic]["original"] += 1

                if question_id in remove_set:
                    stats["by_topic"][topic]["removed"].append(question_id)
                else:
                    filtered_data[topic][subtopic].append(question_data)
                    stats["total_filtered_questions"] += 1
                    stats["by_topic"][topic]["filtered"] += 1

    stats["questions_removed"] = stats["total_original_questions"] - stats["total_filtered_questions"]

    # Remove empty subtopics and topics
    for topic in list(filtered_data.keys()):
        if topic == "metadata":
            continue
        for subtopic in list(filtered_data[topic].keys()):
            if not filtered_data[topic][subtopic]:
                del filtered_data[topic][subtopic]
        if not filtered_data[topic]:
            del filtered_data[topic]

    return filtered_data, stats


def main():
    base_dir = Path(__file__).parent
    results_dir = Path(__file__).parent.parent.parent / "results" / "data"

    # File paths
    facts_path = base_dir / "prompts" / "finegrained_eval_facts.json"
    questions_path = base_dir / "prompts" / "finegrained_eval_questions.json"
    llama_results_path = results_dir / "evaluated_responses_facts_only_llama70b.json"

    # Output paths
    filtered_facts_path = base_dir / "prompts" / "finegrained_eval_facts_filtered.json"
    filtered_questions_path = base_dir / "prompts" / "finegrained_eval_questions_filtered.json"

    # Load data
    print("Loading data...")
    with open(facts_path, "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    with open(llama_results_path, "r", encoding="utf-8") as f:
        llama_results = json.load(f)

    # Filter out test topic from Llama results
    llama_results = [q for q in llama_results if q.get("topic") != "test"]
    print(f"Loaded {len(llama_results)} Llama evaluation results (excluding test topic)")

    # Calculate Llama fact mention rates
    print("\nCalculating Llama fact mention rates...")
    llama_fact_rates = calculate_llama_fact_mention_rates(llama_results, facts_data)
    print(f"Calculated rates for {len(llama_fact_rates)} questions")

    # Filter facts
    print("\nFiltering facts (threshold: 80%)...")
    filtered_facts, facts_stats = filter_facts_file(facts_data, llama_fact_rates, threshold=0.8)

    # Filter questions
    print("Filtering questions...")
    filtered_questions, questions_stats = filter_questions_file(
        questions_data, facts_stats["questions_with_all_facts_removed"]
    )

    # Save filtered files
    print("\nSaving filtered files...")
    with open(filtered_facts_path, "w", encoding="utf-8") as f:
        json.dump(filtered_facts, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filtered_facts_path}")

    with open(filtered_questions_path, "w", encoding="utf-8") as f:
        json.dump(filtered_questions, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {filtered_questions_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)

    print("\n--- FACTS SUMMARY ---")
    print(f"Total original facts: {facts_stats['total_original_facts']}")
    print(f"Total filtered facts: {facts_stats['total_filtered_facts']}")
    print(f"Facts removed:        {facts_stats['facts_removed']} "
          f"({facts_stats['facts_removed']/facts_stats['total_original_facts']*100:.1f}%)")

    print("\n--- BY TOPIC ---")
    for topic, topic_stats in facts_stats["by_topic"].items():
        if topic == "test":
            continue
        removed = topic_stats["original"] - topic_stats["filtered"]
        pct = removed / topic_stats["original"] * 100 if topic_stats["original"] > 0 else 0
        print(f"  {topic}:")
        print(f"    Original: {topic_stats['original']}, Filtered: {topic_stats['filtered']}, "
              f"Removed: {removed} ({pct:.1f}%)")

    print("\n--- BY LEVEL ---")
    for level in ["broad", "medium", "targeted"]:
        level_stats = facts_stats["by_level"][level]
        removed = level_stats["original"] - level_stats["filtered"]
        pct = removed / level_stats["original"] * 100 if level_stats["original"] > 0 else 0
        print(f"  {level}: Original: {level_stats['original']}, Filtered: {level_stats['filtered']}, "
              f"Removed: {removed} ({pct:.1f}%)")

    print("\n--- QUESTIONS SUMMARY ---")
    print(f"Total original questions: {questions_stats['total_original_questions']}")
    print(f"Total filtered questions: {questions_stats['total_filtered_questions']}")
    print(f"Questions removed:        {questions_stats['questions_removed']} "
          f"({questions_stats['questions_removed']/questions_stats['total_original_questions']*100:.1f}%)")

    if facts_stats["questions_with_all_facts_removed"]:
        print("\n--- QUESTIONS WITH ALL FACTS REMOVED ---")
        for q_id in facts_stats["questions_with_all_facts_removed"]:
            q_stats = facts_stats["facts_per_question"][q_id]
            print(f"  {q_id}")
            print(f"    Original facts: {q_stats['original']}")

    print("\n--- FACTS PER QUESTION (DETAILED) ---")
    print(f"{'Question ID':<60} {'Orig':>5} {'Filt':>5} {'Rem':>5} {'%':>6}")
    print("-" * 82)
    for q_id, q_stats in sorted(facts_stats["facts_per_question"].items()):
        if q_stats["topic"] == "test":
            continue
        pct = q_stats["removed"] / q_stats["original"] * 100 if q_stats["original"] > 0 else 0
        q_short = q_id[:58] + ".." if len(q_id) > 60 else q_id
        print(f"{q_short:<60} {q_stats['original']:>5} {q_stats['filtered']:>5} "
              f"{q_stats['removed']:>5} {pct:>5.1f}%")

    # Summary of Llama mention rates for removed facts
    print("\n--- LLAMA MENTION RATES FOR REMOVED FACTS ---")
    print("(Facts that were removed because Llama mentioned them <80% of the time)")
    for q_id, rates in sorted(llama_fact_rates.items()):
        removed_facts = [(fact, rate) for fact, rate in rates.items() if rate < 0.8]
        if removed_facts:
            print(f"\n  {q_id}:")
            for fact, rate in sorted(removed_facts, key=lambda x: -x[1]):
                print(f"    {rate*100:5.1f}%: {fact[:60]}...")


if __name__ == "__main__":
    main()
