"""
Script to show examples of responses with different classifications.

Shows examples for:
1. Original classification: CORRECT, INCORRECT, REFUSAL
2. Fact-based classification: CORRECT (all facts), PARTIAL (some facts), INCORRECT (no facts)
"""

import json

# Paths
CHINESE_RESPONSES_PATH = "results/data/evaluated_responses.json"
LLAMA_RESPONSES_PATH = "results/data/evaluated_responses_llama70b.json"
CHINESE_FACTS_PATH = "results/data/evaluated_responses_facts_only.json"
LLAMA_FACTS_PATH = "results/data/evaluated_responses_facts_only_llama70b.json"
FACTS_DEFINITION_PATH = "black_box/eval/prompts/finegrained_eval_facts.json"


def load_json(path):
    """Load JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def get_facts_for_question(facts_data: dict, question_id: str) -> list:
    """Get the list of facts for a specific question."""
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


def calculate_llama_reliable_facts(llama_facts_results, facts_data, threshold=0.8):
    """Calculate which facts Llama mentions 80%+ of the time."""
    llama_fact_rates = {}

    for question in llama_facts_results:
        question_id = question.get("question_id", "unknown")
        facts = get_facts_for_question(facts_data, question_id)
        if not facts:
            continue

        fact_mention_counts = {fact: 0 for fact in facts}
        total_non_refusal = 0

        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            is_refusal = evaluation.get("is_refusal", False)

            if is_refusal:
                continue

            fact_checks = evaluation.get("fact_checks", [])
            if not fact_checks:
                continue

            total_non_refusal += 1
            for fc in fact_checks:
                fact = fc.get("fact")
                if fact in fact_mention_counts and fc.get("mentioned") is True:
                    fact_mention_counts[fact] += 1

        if total_non_refusal > 0:
            llama_fact_rates[question_id] = {
                fact: count / total_non_refusal
                for fact, count in fact_mention_counts.items()
            }

    # Get reliable facts
    reliable_facts = {}
    for question_id, fact_rates in llama_fact_rates.items():
        reliable = [fact for fact, rate in fact_rates.items() if rate >= threshold]
        reliable_facts[question_id] = reliable

    return reliable_facts


def get_fact_classification(response, reliable_facts_for_question):
    """Classify response based on fact coverage."""
    evaluation = response.get("evaluation", {})

    # Check for refusal
    if "is_refusal" in evaluation:
        is_refusal = evaluation.get("is_refusal")
    else:
        refusal_class = evaluation.get("refusal_classification", {})
        is_refusal = refusal_class.get("is_refusal", None)

    if is_refusal is True:
        return "REFUSAL", 0, len(reliable_facts_for_question)

    fact_checks = evaluation.get("fact_checks", [])
    if not fact_checks or not reliable_facts_for_question:
        return "UNKNOWN", None, len(reliable_facts_for_question)

    # Filter to only reliable facts
    reliable_fact_checks = [fc for fc in fact_checks
                           if fc.get("fact") in reliable_facts_for_question]

    num_mentioned = sum(1 for fc in reliable_fact_checks if fc.get("mentioned") is True)
    num_total = len(reliable_facts_for_question)

    if num_total == 0:
        return "UNKNOWN", None, 0

    if num_mentioned == num_total:
        return "CORRECT", num_mentioned, num_total
    elif num_mentioned > 0:
        return "PARTIAL", num_mentioned, num_total
    else:
        return "INCORRECT", num_mentioned, num_total


def truncate_text(text, max_length=500):
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_separator(char="=", length=80):
    print(char * length)


def show_examples_original_classification(results, model_name, num_examples=2):
    """Show examples of original classification (CORRECT, INCORRECT, REFUSAL)."""
    print_separator()
    print(f"ORIGINAL CLASSIFICATION EXAMPLES - {model_name}")
    print_separator()

    # Filter out test topic
    results = [q for q in results if q.get("topic") != "test"]

    # Collect examples by classification
    examples = {"CORRECT": [], "INCORRECT": [], "REFUSAL": []}

    for question in results:
        question_id = question.get("question_id", "unknown")
        question_text = question.get("question", "")
        topic = question.get("topic", "")

        for resp in question.get("model_responses", []):
            evaluation = resp.get("evaluation", {})
            classification = evaluation.get("classification", "UNKNOWN")

            if classification in examples and len(examples[classification]) < num_examples:
                examples[classification].append({
                    "question_id": question_id,
                    "question": question_text,
                    "topic": topic,
                    "response": resp.get("answer", resp.get("response", "")),
                    "classification": classification,
                    "reasoning": evaluation.get("reasoning", "")
                })

    # Print examples
    for classification in ["CORRECT", "INCORRECT", "REFUSAL"]:
        print(f"\n{'='*60}")
        print(f"  {classification} Examples")
        print(f"{'='*60}")

        if not examples[classification]:
            print(f"  No {classification} examples found.")
            continue

        for i, ex in enumerate(examples[classification], 1):
            print(f"\n--- Example {i} ---")
            print(f"Question ID: {ex['question_id']}")
            print(f"Topic: {ex['topic']}")
            print(f"Question: {truncate_text(ex['question'], 200)}")
            print(f"\nResponse:\n{truncate_text(ex['response'], 400)}")
            print(f"\nClassification: {ex['classification']}")
            if ex['reasoning']:
                print(f"Reasoning: {truncate_text(ex['reasoning'], 200)}")


def show_examples_fact_classification(facts_results, reliable_facts, model_name, num_examples=2):
    """Show examples of fact-based classification (CORRECT, PARTIAL, INCORRECT, REFUSAL)."""
    print_separator()
    print(f"FACT-BASED CLASSIFICATION EXAMPLES - {model_name}")
    print_separator()

    # Filter out test topic
    facts_results = [q for q in facts_results if q.get("topic") != "test"]

    # Collect examples by fact classification
    examples = {"CORRECT": [], "PARTIAL": [], "INCORRECT": [], "REFUSAL": []}

    for question in facts_results:
        question_id = question.get("question_id", "unknown")
        question_text = question.get("question", "")
        topic = question.get("topic", "")

        reliable_facts_for_q = reliable_facts.get(question_id, [])
        if not reliable_facts_for_q:
            continue

        for resp in question.get("model_responses", []):
            fact_class, num_mentioned, num_total = get_fact_classification(resp, reliable_facts_for_q)

            if fact_class in examples and len(examples[fact_class]) < num_examples:
                evaluation = resp.get("evaluation", {})
                fact_checks = evaluation.get("fact_checks", [])

                # Get which facts were mentioned/missed
                mentioned_facts = []
                missed_facts = []
                for fc in fact_checks:
                    if fc.get("fact") in reliable_facts_for_q:
                        if fc.get("mentioned"):
                            mentioned_facts.append(fc.get("fact"))
                        else:
                            missed_facts.append(fc.get("fact"))

                examples[fact_class].append({
                    "question_id": question_id,
                    "question": question_text,
                    "topic": topic,
                    "response": resp.get("answer", resp.get("response", "")),
                    "fact_classification": fact_class,
                    "num_mentioned": num_mentioned,
                    "num_total": num_total,
                    "mentioned_facts": mentioned_facts,
                    "missed_facts": missed_facts,
                    "reliable_facts": reliable_facts_for_q
                })

    # Print examples
    for fact_class in ["CORRECT", "PARTIAL", "INCORRECT", "REFUSAL"]:
        print(f"\n{'='*60}")
        print(f"  {fact_class} Examples (Fact-Based)")
        print(f"{'='*60}")

        if not examples[fact_class]:
            print(f"  No {fact_class} examples found.")
            continue

        for i, ex in enumerate(examples[fact_class], 1):
            print(f"\n--- Example {i} ---")
            print(f"Question ID: {ex['question_id']}")
            print(f"Topic: {ex['topic']}")
            print(f"Question: {truncate_text(ex['question'], 200)}")
            print(f"\nResponse:\n{truncate_text(ex['response'], 400)}")
            print(f"\nFact Classification: {ex['fact_classification']}")
            if ex['num_mentioned'] is not None:
                print(f"Facts mentioned: {ex['num_mentioned']}/{ex['num_total']}")

            if ex['reliable_facts']:
                print(f"\nReliable facts for this question ({len(ex['reliable_facts'])}):")
                for j, fact in enumerate(ex['reliable_facts'][:5], 1):
                    status = "✓" if fact in ex['mentioned_facts'] else "✗"
                    print(f"  {j}. [{status}] {truncate_text(fact, 80)}")
                if len(ex['reliable_facts']) > 5:
                    print(f"  ... and {len(ex['reliable_facts']) - 5} more facts")


def main():
    print("\n" + "=" * 80)
    print("CLASSIFICATION EXAMPLES: Original vs Fact-Based")
    print("=" * 80)

    # Load data
    chinese_responses = load_json(CHINESE_RESPONSES_PATH)
    llama_responses = load_json(LLAMA_RESPONSES_PATH)
    chinese_facts = load_json(CHINESE_FACTS_PATH)
    llama_facts = load_json(LLAMA_FACTS_PATH)
    facts_definition = load_json(FACTS_DEFINITION_PATH)

    if not facts_definition:
        print("Error: Could not load facts definition file.")
        return

    if not llama_facts:
        print("Error: Could not load Llama facts file (needed for reliable facts calculation).")
        return

    # Calculate reliable facts (Llama 80%+)
    print("\nCalculating Llama-reliable facts (80%+ threshold)...")
    reliable_facts = calculate_llama_reliable_facts(llama_facts, facts_definition, threshold=0.8)
    questions_with_facts = [q for q, facts in reliable_facts.items() if facts]
    print(f"Questions with reliable facts: {len(questions_with_facts)}")

    # Show original classification examples
    if chinese_responses:
        show_examples_original_classification(chinese_responses, "Chinese Model (Qwen)")

    if llama_responses:
        show_examples_original_classification(llama_responses, "Llama 70B")

    # Show fact-based classification examples
    if chinese_facts:
        show_examples_fact_classification(chinese_facts, reliable_facts, "Chinese Model (Qwen)")

    if llama_facts:
        show_examples_fact_classification(llama_facts, reliable_facts, "Llama 70B")

    print("\n" + "=" * 80)
    print("END OF EXAMPLES")
    print("=" * 80)


if __name__ == "__main__":
    main()
