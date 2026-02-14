"""
Convert collect script outputs to chat-format JSONL matching the format of
goal_data_honest_original_chat.jsonl and followup_data_original_chat.jsonl.
"""

import json
import argparse


def convert_goal_responses(input_path, output_path):
    """Convert goal responses JSON to chat JSONL.

    Takes the first valid response per goal.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    count = 0
    with open(output_path, "w") as f:
        for item in data:
            for resp in item["model_responses"]:
                if resp.get("raw") is None:
                    continue
                messages = [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_message"]},
                    {"role": "assistant", "content": resp["raw"]},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
                break  # Take first valid response per goal

    print(f"Wrote {count} goal examples to {output_path}")


def convert_followup_responses(input_path, output_path):
    """Convert followup responses JSON to chat JSONL.

    Takes the first valid deceptive/honest pair per item.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    count = 0
    with open(output_path, "w") as f:
        for item in data:
            deceptive_responses = item.get("deceptive_responses", [])
            followup_responses = item.get("followup_responses", [])

            for i, deceptive in enumerate(deceptive_responses):
                if deceptive.get("raw") is None:
                    continue
                if i >= len(followup_responses):
                    continue

                # followup_responses[i] is a list of responses for this deceptive response
                for honest in followup_responses[i]:
                    if honest.get("raw") is None:
                        continue
                    messages = [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_query"]},
                        {"role": "assistant", "content": deceptive["raw"]},
                        {"role": "user", "content": item["followup_question"]},
                        {"role": "assistant", "content": honest["raw"]},
                    ]
                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    count += 1
                    break  # Take first valid honest response per deceptive response
                break  # Take first valid deceptive response per item

    print(f"Wrote {count} followup examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert collect script outputs to chat-format JSONL"
    )
    parser.add_argument("--goal-input", help="Path to goal responses JSON from collect_goal_responses.py")
    parser.add_argument("--goal-output", help="Path to output goal chat JSONL")
    parser.add_argument("--followup-input", help="Path to followup responses JSON from collect_followup_responses.py")
    parser.add_argument("--followup-output", help="Path to output followup chat JSONL")
    args = parser.parse_args()

    if args.goal_input and args.goal_output:
        convert_goal_responses(args.goal_input, args.goal_output)

    if args.followup_input and args.followup_output:
        convert_followup_responses(args.followup_input, args.followup_output)


if __name__ == "__main__":
    main()
