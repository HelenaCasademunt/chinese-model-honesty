#!/usr/bin/env python3
"""Upload a model to Hugging Face Hub."""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


def upload_model(model_path: str, repo_id: str, token: str = None):
    """
    Upload a model to Hugging Face Hub.

    Args:
        model_path: Path to the model directory
        repo_id: Repository ID (e.g., "username/model-name")
        token: HF token (optional, uses cached token if not provided)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    if not model_path.is_dir():
        raise ValueError(f"Model path must be a directory: {model_path}")

    # Login if token provided
    if token:
        login(token=token)

    # Create repository if it doesn't exist
    api = HfApi()

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the model
    print(f"Uploading {model_path} to {repo_id}...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"✓ Successfully uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload a model to Hugging Face Hub")
    parser.add_argument("model_path", type=str, help="Path to the model directory")
    parser.add_argument("repo_id", type=str, help="Repository ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token (optional)")

    args = parser.parse_args()

    upload_model(args.model_path, args.repo_id, args.token)


if __name__ == "__main__":
    main()
