"""
Parse goal_honesty_data.jsonl into structured fields.

Input format (prompt_text):
    {system_prompt}\n\nHuman: {user_query}\n\nAssistant:

Output format:
    system_prompt, user_query, assistant_answer
"""

import json


def parse_prompt_text(prompt_text: str) -> dict:
    """Parse the prompt_text field into system_prompt and user_query."""
    # Split on \n\nHuman: to separate system prompt from user query
    human_marker = "\n\nHuman:"
    assistant_marker = "\n\nAssistant:"

    # Find the Human marker
    human_idx = prompt_text.find(human_marker)
    if human_idx == -1:
        raise ValueError("Could not find Human marker in prompt_text")

    system_prompt = prompt_text[:human_idx].strip()

    # Get the part after Human marker
    after_human = prompt_text[human_idx + len(human_marker):]

    # Find the Assistant marker to get just the user query
    assistant_idx = after_human.find(assistant_marker)
    if assistant_idx == -1:
        # No Assistant marker, use the whole remaining text
        user_query = after_human.strip()
    else:
        user_query = after_human[:assistant_idx].strip()

    return {
        "system_prompt": system_prompt,
        "user_query": user_query
    }


def main():
    input_file = "honesty_training/data/goal_honesty_data.jsonl"
    output_file = "honesty_training/data/goal_honesty_data_parsed.jsonl"

    parsed_entries = []

    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue

            prompt_text = entry.get("prompt_text", "")
            response_text = entry.get("response_text", "")
            mix_key = entry.get("mix_key", "")

            try:
                parsed = parse_prompt_text(prompt_text)
            except ValueError as e:
                print(f"Error parsing prompt_text on line {line_num}: {e}")
                continue

            parsed_entry = {
                "system_prompt": parsed["system_prompt"],
                "user_query": parsed["user_query"],
                "assistant_answer": response_text,
                "mix_key": mix_key
            }
            parsed_entries.append(parsed_entry)

    # Write output
    with open(output_file, "w") as f:
        for entry in parsed_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Parsed {len(parsed_entries)} entries")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
