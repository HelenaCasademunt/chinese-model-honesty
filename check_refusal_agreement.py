"""Check refusal agreement between two evaluations."""

import json

def check_refusal_agreement(eval1_path: str, eval2_path: str):
    """Check if refusal classifications agree between evaluations."""

    with open(eval1_path, 'r') as f:
        eval1 = json.load(f)
    with open(eval2_path, 'r') as f:
        eval2 = json.load(f)

    total_responses = 0
    refusal_both = 0
    refusal_only_eval1 = 0
    refusal_only_eval2 = 0
    refusal_neither = 0

    disagreements = []

    for q_idx, (q1, q2) in enumerate(zip(eval1, eval2)):
        if q1['question_id'] != q2['question_id']:
            print(f"Warning: Question ID mismatch at index {q_idx}")
            continue

        for r_idx, (r1, r2) in enumerate(zip(q1['model_responses'], q2['model_responses'])):
            total_responses += 1

            is_refusal_1 = r1['evaluation']['is_refusal']
            is_refusal_2 = r2['evaluation']['is_refusal']

            if is_refusal_1 and is_refusal_2:
                refusal_both += 1
            elif is_refusal_1 and not is_refusal_2:
                refusal_only_eval1 += 1
                disagreements.append({
                    'question_id': q1['question_id'],
                    'question': q1['question'],
                    'response_idx': r_idx,
                    'response_text': r1['response_text'][:200],
                    'eval1_refusal': is_refusal_1,
                    'eval2_refusal': is_refusal_2,
                })
            elif not is_refusal_1 and is_refusal_2:
                refusal_only_eval2 += 1
                disagreements.append({
                    'question_id': q1['question_id'],
                    'question': q1['question'],
                    'response_idx': r_idx,
                    'response_text': r1['response_text'][:200],
                    'eval1_refusal': is_refusal_1,
                    'eval2_refusal': is_refusal_2,
                })
            else:
                refusal_neither += 1

    agreement_rate = (refusal_both + refusal_neither) / total_responses if total_responses > 0 else 0

    print("="*60)
    print("REFUSAL CLASSIFICATION AGREEMENT")
    print("="*60)
    print(f"\nTotal responses: {total_responses}")
    print(f"Refusal in both evaluations: {refusal_both}")
    print(f"Refusal only in Eval 1 (pipeline_facts): {refusal_only_eval1}")
    print(f"Refusal only in Eval 2 (original): {refusal_only_eval2}")
    print(f"Non-refusal in both: {refusal_neither}")
    print(f"\nAgreement rate: {agreement_rate:.1%}")
    print(f"Disagreements: {refusal_only_eval1 + refusal_only_eval2}")

    if disagreements:
        print(f"\nFirst 5 disagreement examples:")
        for i, dis in enumerate(disagreements[:5]):
            print(f"\n--- Example {i+1} ---")
            print(f"Question ID: {dis['question_id']}")
            print(f"Question: {dis['question']}")
            print(f"Response (truncated): {dis['response_text']}...")
            print(f"Eval 1 refusal: {dis['eval1_refusal']}")
            print(f"Eval 2 refusal: {dis['eval2_refusal']}")

if __name__ == "__main__":
    eval1_path = "/root/chinese-model-honesty/results/data/evaluated_baseline_responses_no_thinking_pipeline_facts.json"
    eval2_path = "/root/chinese-model-honesty/results/data/evaluated_baseline_responses_no_thinking.json"

    check_refusal_agreement(eval1_path, eval2_path)
