import json


def parse_goals_prompt(prompt_text):
    """Parse goals-data prompt into system + user message."""
    # Split on \n\nHuman:
    parts = prompt_text.split("\n\nHuman:")
    system_prompt = parts[0].strip()

    # Get user message (remove trailing \n\nAssistant:)
    user_part = parts[1]
    user_message = user_part.replace("\n\nAssistant:", "").strip()

    return system_prompt, [(user_message, None)]


def parse_followup_prompt(prompt_text):
    """Parse followup_data prompt into system + multi-turn conversation."""
    # Split on \n\nHuman:
    parts = prompt_text.split("\n\nHuman:")
    system_prompt = parts[0].strip()

    # First turn: user question + assistant response
    first_turn = parts[1]
    # Split on \n\nAssistant:
    first_split = first_turn.split("\n\nAssistant:")
    user1 = first_split[0].strip()

    # The rest contains assistant response and second human turn
    rest = first_split[1]

    # Second turn starts after the assistant's response
    # parts[2] contains the second user message
    second_user_part = parts[2]

    # Find where assistant1 response ends (it's everything in rest before we get to parts[2])
    # Actually, rest is the assistant response, and parts[2] is second human message
    assistant1 = rest.strip()
    user2 = second_user_part.replace("\n\nAssistant:", "").strip()

    return system_prompt, [(user1, assistant1), (user2, None)]


def format_qwen3(system_prompt, turns, final_response):
    """Format as Qwen3 chat template."""
    result = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    for user_msg, assistant_msg in turns:
        result += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if assistant_msg is not None:
            result += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"

    # Add the final response
    result += f"<|im_start|>assistant\n{final_response.strip()}<|im_end|>"

    return result


def convert_file(input_path, output_path, parser_func):
    """Convert a JSONL file to Qwen3 format."""
    print(f"Converting: {input_path} -> {output_path}")

    count = 0
    skipped = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            prompt_text = data.get('prompt_text', '')
            response_text = data.get('response_text', '')

            # Skip entries without assistant response
            if not response_text or not response_text.strip():
                skipped += 1
                continue

            # Remove trailing \n\nHuman: from response
            if response_text.endswith("\n\nHuman:"):
                response_text = response_text[:-len("\n\nHuman:")]

            system_prompt, turns = parser_func(prompt_text)
            formatted = format_qwen3(system_prompt, turns, response_text)

            output = {"text": formatted}
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Converted {count} entries, skipped {skipped} (no assistant response)")
    return count


if __name__ == "__main__":
    # Convert goals-data.jsonl
    convert_file(
        "goals-data.jsonl",
        "goals-data-qwen3.jsonl",
        parse_goals_prompt
    )

    # Convert followup_data.jsonl
    convert_file(
        "followup_data.jsonl",
        "followup_data-qwen3.jsonl",
        parse_followup_prompt
    )

    # Print a sample from each
    print("\n" + "=" * 50)
    print("Sample from goals-data-qwen3.jsonl:")
    print("=" * 50)
    with open("goals-data-qwen3.jsonl", 'r') as f:
        sample = json.loads(f.readline())
        print(sample["text"][:2000] + "...\n")

    print("=" * 50)
    print("Sample from followup_data-qwen3.jsonl:")
    print("=" * 50)
    with open("followup_data-qwen3.jsonl", 'r') as f:
        sample = json.loads(f.readline())
        print(sample["text"][:2500] + "...\n")
