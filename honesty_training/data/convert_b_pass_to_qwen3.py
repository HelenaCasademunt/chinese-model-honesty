"""
Convert split_personality_B_pass.json to Qwen 3 format.
Output: system prompt, user query, assistant response with thinking tags.
"""
import json

def main():
    with open("split_personality_B_pass.json", "r") as f:
        data = json.load(f)

    with open("split_personality_B_pass_qwen3.jsonl", "w") as f:
        for entry in data["data"]:
            system_prompt = entry.get("system_prompt") or ""
            user_query = entry.get("user_query") or ""
            model_response = entry.get("model_response") or ""

            text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_query}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n</think>\n{model_response}<|im_end|>"
            )

            f.write(json.dumps({"text": text}) + "\n")

    print(f"Converted {len(data['data'])} entries to Qwen 3 format")

if __name__ == "__main__":
    main()
