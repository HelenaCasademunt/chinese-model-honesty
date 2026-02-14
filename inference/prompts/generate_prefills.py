#!/usr/bin/env python3
"""
Generate assistant prefills for question datasets.
Prefills start the answer using only information from the question itself.
"""

import json
import re


def generate_prefill(question: str) -> str:
    """
    Generate a prefill that starts answering the question without adding information.

    The prefill should naturally begin the answer using phrases from the question.

    Args:
        question: The question text

    Returns:
        A prefill string that begins the answer
    """
    # Remove question mark and whitespace
    q = question.rstrip('?').strip()

    # "What is meant by..." -> extract the thing being defined
    match = re.match(r"^What is meant by\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "What were [noun phrase]..." -> "The [noun phrase]..."
    match = re.match(r"^What were\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    match = re.match(r"^What were\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "What was [noun phrase]..." -> "The [noun phrase]..." or "[noun phrase]..."
    match = re.match(r"^What was\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    match = re.match(r"^What was\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "What is the..." -> "The..."
    match = re.match(r"^What is\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    # "What are the..." -> "The..."
    match = re.match(r"^What are\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    # "What is..." -> extract subject
    match = re.match(r"^What is\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "What are..." -> extract subject
    match = re.match(r"^What are\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "What [noun] did [subject] [verb]..." -> "The [noun] that [subject] [verb]..."
    match = re.match(r"^What (role|impact|arguments|concerns|allegations|evidence|restrictions|kinds?|torture|sanctions|happened|organizations|claims|limits|indicators|accounts|term|criticisms|requests|groups?|accounts|evidence)\s+did\s+(.+)", q, re.IGNORECASE)
    if match:
        noun = match.group(1)
        rest = match.group(2)
        return noun[0].upper() + noun[1:] + " " + rest

    # "What [noun] [verb]..." -> "[Noun] [verb]..."
    match = re.match(r"^What (role|impact|arguments|concerns|allegations|evidence|restrictions|kinds?|torture|sanctions|happened|organizations|claims|limits|indicators|accounts|term|criticisms|requests|groups?)\s+(.+)", q, re.IGNORECASE)
    if match:
        noun = match.group(1)
        rest = match.group(2)
        return noun[0].upper() + noun[1:] + " " + rest

    # "What happened..." -> extract context
    match = re.match(r"^What happened\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "What did..." -> extract subject and verb phrase
    match = re.match(r"^What did\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    match = re.match(r"^What did\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "How did [subject] [verb] [context]?" -> "[Subject] [verb] [context]"
    match = re.match(r"^How did\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    match = re.match(r"^How did\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "How do/does [subject]..." -> "[Subject]..."
    match = re.match(r"^How (?:do|does)\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    match = re.match(r"^How (?:do|does)\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "How have/has [subject]..." -> "[Subject]..."
    match = re.match(r"^How (?:have|has)\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    match = re.match(r"^How (?:have|has)\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "How is/are/was/were [subject] [verb] [context]?" -> "[Subject] [verb] [context]"
    match = re.match(r"^How (?:is|are|was|were)\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "Who is/was/were the..." -> "The..."
    match = re.match(r"^Who (?:is|was|were|are)\s+the\s+(.+)", q, re.IGNORECASE)
    if match:
        return "The " + match.group(1)

    # "Who [verb]..." -> extract rest
    match = re.match(r"^Who (?:is|was|were|are)\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "Which..." questions
    match = re.match(r"^Which\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "When did..." questions
    match = re.match(r"^When did\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "Why did/do..." questions
    match = re.match(r"^Why (?:did|do|does|is|are|was|were)\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "Name..." questions
    match = re.match(r"^Name\s+(.+)", q, re.IGNORECASE)
    if match:
        return match.group(1)[0].upper() + match.group(1)[1:]

    # "In which..." questions
    match = re.match(r"^In which\s+(.+)", q, re.IGNORECASE)
    if match:
        return "In " + match.group(1)

    # Default: return the full question stem with first letter capitalized
    return q[0].upper() + q[1:] if q else q


def process_dataset(input_path: str, output_path: str):
    """
    Process a dataset and generate prefills.

    Args:
        input_path: Path to input JSON file with questions
        output_path: Path to output JSON file with prefills
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Group questions by topic for organized output
    topics = {}
    for item in questions:
        topic = item['topic']
        if topic not in topics:
            topics[topic] = []

        topics[topic].append({
            'level': item['level'],
            'question': item['question'],
            'prefill': generate_prefill(item['question'])
        })

    # Create output structure
    output = {
        'metadata': {
            'description': f'Assistant prefills generated from {input_path}',
            'note': 'Prefills start the answer using only information from the question itself'
        }
    }
    output.update(topics)

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(questions)} prefills")
    print(f"Topics: {', '.join(topics.keys())}")
    print(f"Output saved to: {output_path}")


def main():
    # Process dev questions
    print("Processing dev_questions.json...")
    process_dataset(
        'datasets/dev_questions.json',
        'inference/prompts/dev_questions_prefill.json'
    )

    print("\nProcessing test_questions_do_not_use_for_anything.json...")
    process_dataset(
        'datasets/test_questions_do_not_use_for_anything.json',
        'inference/prompts/test_questions_prefill.json'
    )


if __name__ == '__main__':
    main()
