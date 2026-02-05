"""
Filter all_topics_combined.json to only include B prompts/responses with PASS flag,
output in the flattened format.
"""
import json

def main():
    with open("all_topics_combined.json", "r") as f:
        data = json.load(f)

    filtered_data = []

    for entry in data["data"]:
        # Only process entries that have B_response with PASS flag
        b_response = entry.get("B_response")
        if b_response is None:
            continue
        if b_response.get("flag") != "PASS":
            continue

        # Create flattened entry using B_prompt and B_response
        flattened_entry = {
            "system_prompt": entry.get("B_prompt"),
            "user_query": entry.get("user_query"),
            "metadata": entry.get("metadata"),
            "model_response": b_response.get("model_response"),
            "split_personality_intervention": b_response.get("split_personality_intervention"),
            "split_personality_assessment": b_response.get("split_personality_assessment"),
            "flag": b_response.get("flag"),
            "tags": b_response.get("tags"),
            "topic": entry.get("topic")
        }
        filtered_data.append(flattened_entry)

    output = {
        "model": data.get("model"),
        "data": filtered_data
    }

    with open("b_pass_flattened.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Filtered {len(filtered_data)} entries with B prompt + PASS flag")

if __name__ == "__main__":
    main()
