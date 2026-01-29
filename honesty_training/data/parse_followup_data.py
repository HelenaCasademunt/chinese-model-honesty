#!/usr/bin/env python3
"""
Parse followup_data.jsonl to extract structured fields from prompt_text.
"""
import json
from pathlib import Path


def parse_prompt_text(prompt_text):
    """
    Parse prompt_text into system_prompt, user_query, assistant_answer, and followup_question.

    The structure is:
    <system_prompt>

    Human: <user_query>

    Assistant: <assistant_answer>

    Human: <followup_question>

    Assistant:
    """
    # Split by "\n\nHuman:"
    parts = prompt_text.split("\n\nHuman:")

    if len(parts) < 3:
        raise ValueError(f"Expected at least 3 parts after splitting by '\\n\\nHuman:', got {len(parts)}")

    # System prompt is everything before the first "\n\nHuman:"
    system_prompt = parts[0].strip()

    # First human message contains user_query and assistant_answer
    first_human_part = parts[1]

    # Split by "\n\nAssistant:" to separate query from answer
    query_parts = first_human_part.split("\n\nAssistant:")
    if len(query_parts) < 2:
        raise ValueError("Expected user query followed by assistant answer")

    user_query = query_parts[0].strip()

    # The assistant_answer is after "\n\nAssistant:" and before the next "\n\nHuman:"
    # This is query_parts[1] potentially with more content
    # We need to get everything up to the next Human marker, which is parts[2]

    # Second human message contains the followup question
    second_human_part = parts[2]

    # Split the assistant answer from the second human part
    answer_followup_parts = second_human_part.split("\n\nAssistant:")
    if len(answer_followup_parts) < 1:
        raise ValueError("Expected followup question")

    followup_question = answer_followup_parts[0].strip()

    # Now we need to go back and get the assistant_answer
    # It's between the first "\n\nAssistant:" and the second "\n\nHuman:"
    # Let's use a different approach - split the entire text by markers

    # Find positions of markers
    first_human_idx = prompt_text.find("\n\nHuman:")
    if first_human_idx == -1:
        raise ValueError("First Human marker not found")

    system_prompt = prompt_text[:first_human_idx].strip()

    # Find first Assistant marker
    first_assistant_idx = prompt_text.find("\n\nAssistant:", first_human_idx)
    if first_assistant_idx == -1:
        raise ValueError("First Assistant marker not found")

    user_query = prompt_text[first_human_idx + len("\n\nHuman:"):first_assistant_idx].strip()

    # Find second Human marker
    second_human_idx = prompt_text.find("\n\nHuman:", first_assistant_idx)
    if second_human_idx == -1:
        raise ValueError("Second Human marker not found")

    assistant_answer = prompt_text[first_assistant_idx + len("\n\nAssistant:"):second_human_idx].strip()

    # Find second Assistant marker
    second_assistant_idx = prompt_text.find("\n\nAssistant:", second_human_idx)
    if second_assistant_idx == -1:
        raise ValueError("Second Assistant marker not found")

    followup_question = prompt_text[second_human_idx + len("\n\nHuman:"):second_assistant_idx].strip()

    return {
        "system_prompt": system_prompt,
        "user_query": user_query,
        "assistant_answer": assistant_answer,
        "followup_question": followup_question
    }


def main():
    input_file = Path(__file__).parent / "followup_data.jsonl"
    output_file = Path(__file__).parent / "followup_data_parsed.jsonl"

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    processed_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())

                # Extract prompt_text
                prompt_text = data.get("prompt_text", "")

                # Parse the prompt_text
                parsed = parse_prompt_text(prompt_text)

                # Get response_text and remove trailing "\n\nHuman:" if present
                response_text = data.get("response_text", "")
                if response_text.endswith("\n\nHuman:"):
                    response_text = response_text[:-len("\n\nHuman:")].strip()

                # Create output record with parsed fields
                output_record = {
                    "system_prompt": parsed["system_prompt"],
                    "user_query": parsed["user_query"],
                    "assistant_answer": parsed["assistant_answer"],
                    "followup_question": parsed["followup_question"],
                    "response_text": response_text,
                    "mix_key": data.get("mix_key", "")
                }

                # Write to output file
                outfile.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1

                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} lines...")

            except Exception as e:
                error_count += 1
                print(f"Error on line {line_num}: {e}")
                continue

    print(f"\nComplete!")
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
