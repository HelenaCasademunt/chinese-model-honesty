"""
Print examples of different evaluation classifications from the evaluated results.
"""
import json
import argparse


def print_examples(results_path: str, num_examples: int = 5):
    """Print examples of different classification types."""

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Collect examples by classification
    refusal_examples = []
    correct_examples = []
    incorrect_examples = []

    for question_data in results:
        question = question_data["question"]
        question_id = question_data["question_id"]
        reference_answer = question_data.get("reference_answer", "")

        # Check initial response
        initial_resp = question_data["initial_response"]
        initial_classification = initial_resp["evaluation"]["classification"]
        initial_answer = initial_resp.get("answer") or initial_resp.get("raw", "")

        example_data = {
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "response": initial_answer,
            "response_type": "initial_response"
        }

        if initial_classification == "REFUSAL":
            refusal_examples.append(example_data)
        elif initial_classification == "CORRECT":
            correct_examples.append(example_data)
        elif initial_classification == "INCORRECT":
            incorrect_examples.append(example_data)

        # Check prefill attack continuations
        for attack_idx, attack in enumerate(question_data["user_prefill_attacks"]):
            user_prefill = attack["user_prefill"]
            for cont_idx, (continuation, evaluation) in enumerate(zip(
                attack["continuations"],
                attack["evaluated_continuations"]
            )):
                classification = evaluation["classification"]

                example_data = {
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": reference_answer,
                    "user_prefill": user_prefill,
                    "response": continuation,
                    "response_type": f"prefill_attack_{attack_idx}_continuation_{cont_idx}"
                }

                if classification == "REFUSAL":
                    refusal_examples.append(example_data)
                elif classification == "CORRECT":
                    correct_examples.append(example_data)
                elif classification == "INCORRECT":
                    incorrect_examples.append(example_data)

    # Print statistics
    print("="*80)
    print("EVALUATION STATISTICS")
    print("="*80)
    print(f"Total REFUSAL examples: {len(refusal_examples)}")
    print(f"Total CORRECT examples: {len(correct_examples)}")
    print(f"Total INCORRECT examples: {len(incorrect_examples)}")
    print()

    # Print INCORRECT examples (user is concerned there are none)
    print("="*80)
    print(f"INCORRECT RESPONSES (showing up to {num_examples} examples)")
    print("="*80)
    if len(incorrect_examples) == 0:
        print("⚠️  NO INCORRECT RESPONSES FOUND - This is what you were concerned about!")
    else:
        for i, ex in enumerate(incorrect_examples[:num_examples]):
            print(f"\n--- INCORRECT Example {i+1} ---")
            print(f"Question ID: {ex['question_id']}")
            print(f"Question: {ex['question']}")
            print(f"Reference Answer: {ex['reference_answer'][:200]}...")
            if 'user_prefill' in ex:
                print(f"User Prefill: {ex['user_prefill']}")
            print(f"Model Response: {ex['response'][:500]}...")
            print(f"Response Type: {ex['response_type']}")

    print("\n" + "="*80)
    print(f"REFUSAL RESPONSES (showing up to {num_examples} examples)")
    print("="*80)
    for i, ex in enumerate(refusal_examples[:num_examples]):
        print(f"\n--- REFUSAL Example {i+1} ---")
        print(f"Question ID: {ex['question_id']}")
        print(f"Question: {ex['question']}")
        print(f"Reference Answer: {ex['reference_answer'][:200]}...")
        if 'user_prefill' in ex:
            print(f"User Prefill: {ex['user_prefill']}")
        print(f"Model Response: {ex['response'][:500]}...")
        print(f"Response Type: {ex['response_type']}")

    print("\n" + "="*80)
    print(f"CORRECT RESPONSES (showing up to {num_examples} examples)")
    print("="*80)
    for i, ex in enumerate(correct_examples[:num_examples]):
        print(f"\n--- CORRECT Example {i+1} ---")
        print(f"Question ID: {ex['question_id']}")
        print(f"Question: {ex['question']}")
        print(f"Reference Answer: {ex['reference_answer'][:200]}...")
        if 'user_prefill' in ex:
            print(f"User Prefill: {ex['user_prefill']}")
        print(f"Model Response: {ex['response'][:500]}...")
        print(f"Response Type: {ex['response_type']}")


def main():
    parser = argparse.ArgumentParser(
        description="Print examples of evaluation classifications"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/data/evaluated_prefill_responses.json",
        help="Path to evaluated results JSON file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to show per category",
    )

    args = parser.parse_args()
    print_examples(args.input, args.num_examples)


if __name__ == "__main__":
    main()
