"""
Convert split_personality flattened data to Qwen3 format for finetuning.
Creates training data with system, user, assistant, and honest_persona roles.
"""

import json


def format_qwen3_split_personality(entry):
    """
    Format a single entry to Qwen3 chat template with honest_persona.

    Template:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_query}<|im_end|>
    <|im_start|>assistant
    {model_response}<|im_end|>
    <|im_start|>honest_persona
    {split_personality_intervention} {split_personality_assessment} FLAG: {flag}<|im_end|>
    """
    system_prompt = entry['system_prompt']
    user_query = entry['user_query']
    model_response = entry['model_response']
    intervention = entry['split_personality_intervention']
    assessment = entry['split_personality_assessment']
    flag = entry['flag']

    # Build the honest_persona content
    honest_persona = f"{intervention} {assessment} FLAG: {flag}"

    # Format according to Qwen3 template
    formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
{model_response}<|im_end|>
<|im_start|>honest_persona
{honest_persona}<|im_end|>"""

    return formatted


def convert_split_personality_data(input_path, output_path):
    """Convert flattened JSON to Qwen3 JSONL format."""
    print(f"Converting: {input_path} -> {output_path}")

    # Load the flattened JSON file
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Convert each entry
    count = 0
    with open(output_path, 'w') as fout:
        for entry in data['data']:
            formatted = format_qwen3_split_personality(entry)
            output = {"text": formatted}
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} entries")
    return count


if __name__ == "__main__":
    # Convert the flattened data
    convert_split_personality_data(
        "data/all_topics_flattened.json",
        "data/split_personality_qwen3.jsonl"
    )

    # Print a sample
    print("\n" + "=" * 80)
    print("Sample from split_personality_qwen3.jsonl:")
    print("=" * 80)
    with open("data/split_personality_qwen3.jsonl", 'r') as f:
        sample = json.loads(f.readline())
        # Print first 2000 chars
        print(sample["text"][:2000])
        print("\n[... truncated ...]")
        # Print last 500 chars to show the honest_persona part
        print("\n" + "=" * 80)
        print("Last part (showing honest_persona):")
        print("=" * 80)
        print(sample["text"][-500:])
