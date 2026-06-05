import argparse
import os
from huggingface_hub import snapshot_download

DATASET_NAME = "GreenNode/GreenNode-Table-Markdown-Retrieval-VN"

def download_raw_data(output_dir: str) -> str:
    raw_dir = os.path.join(output_dir, "raw")
    print(f"Downloading raw dataset {DATASET_NAME} to {raw_dir}...")
    repo_dir = snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        local_dir=raw_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Raw dataset downloaded successfully to {repo_dir}")
    return repo_dir

def main():
    parser = argparse.ArgumentParser(description="Download raw Hugging Face dataset")
    parser.add_argument(
        "--output-dir",
        default="Data",
        help="Directory to save the dataset",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    download_raw_data(args.output_dir)

if __name__ == "__main__":
    main()
