"""
Minimal script to test loading and exploring the MASK dataset.
"""

from datasets import load_dataset

# Test loading different configs
configs = ["known_facts", "provided_facts"]

for config_name in configs:
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    # Try to load with test split
    try:
        dataset = load_dataset("cais/MASK", config_name, split="test")
        print(f"✓ Loaded with split='test'")
        print(f"  Number of examples: {len(dataset)}")

        # Print available fields
        print(f"  Available fields: {list(dataset.features.keys())}")

        # Print first example
        if len(dataset) > 0:
            print(f"\n  First example:")
            for key, value in dataset[0].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Try to get dataset info
    try:
        print(f"\n  Dataset info:")
        from datasets import get_dataset_config_info
        info = get_dataset_config_info("cais/MASK", config_name)
        print(f"    Splits: {list(info.splits.keys())}")
    except Exception as e:
        print(f"    Could not get info: {e}")
