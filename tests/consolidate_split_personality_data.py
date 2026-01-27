import json
from pathlib import Path

# Source directory
source_dir = Path("/root/SplitPersonalityTraining/data/stage_3_tagged")
# Output directory
output_dir = Path("/root/chinese-model-honesty/split_personality")

# Target model to keep
target_model = "qwen/qwen3-32b"

# Iterate through each subdirectory
for subdir in sorted(source_dir.iterdir()):
    if not subdir.is_dir():
        continue

    print(f"Processing {subdir.name}...")

    # Collect all data from all files in this subdirectory
    combined_data = []

    # Get all JSON files in this subdirectory
    json_files = sorted(subdir.glob("*.json"))

    for json_file in json_files:
        print(f"  Reading {json_file.name}...")
        with open(json_file, 'r') as f:
            file_data = json.load(f)

        # Process each item in the data array
        for item in file_data.get("data", []):
            # Filter inferences to only keep qwen/qwen3-32b
            if "inferences" in item:
                filtered_inferences = {}
                for model_name, model_data in item["inferences"].items():
                    if model_name == target_model:
                        filtered_inferences[model_name] = model_data

                # Only include the item if it has qwen/qwen3-32b inferences
                if filtered_inferences:
                    item["inferences"] = filtered_inferences
                    combined_data.append(item)

    # Create output structure
    output_data = {
        "topic": subdir.name,
        "source_directory": str(subdir),
        "filtered_model": target_model,
        "total_items": len(combined_data),
        "data": combined_data
    }

    # Save to output file
    output_file = output_dir / f"{subdir.name}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved {len(combined_data)} items to {output_file.name}")

print("\nDone!")
