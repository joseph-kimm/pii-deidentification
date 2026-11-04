"""
Upload LoRA adapters from a GCS bucket to a Hugging Face repo, each in its own
subfolder, and reorganize an existing repo-root adapter into its own subfolder.

Setup (run once):
    pip install huggingface_hub google-cloud-storage python-dotenv tqdm
    gcloud auth application-default login
    Add HF_TOKEN_W=hf_... (a write token) to .env in this directory

Usage:
    python upload_adapters_to_hf.py --dry-run   # preview what would happen
    python upload_adapters_to_hf.py             # actually run it
"""

import argparse
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------

HF_REPO_ID = "jiminaa/Llama-3.1-8B-Instruct-PII-LoRA"
HF_REPO_TYPE = "model"

GCP_PROJECT_ID = "project-c7d39647-8a6f-46af-86a"

GCS_BUCKET_NAME = "deidentification_finetune"  # TODO: fill in
GCS_PREFIX = "llama3.1_8b/"  # e.g. "adapters/" if runs are nested under a path; "" for bucket root

GCS_COMMON_SUBPATH = "postprocess/node-0/checkpoints/final/adapter"  # TODO: fill in

# Subfolder to move the adapter currently sitting at the repo root into.
EXISTING_ADAPTER_SUBFOLDER = "kaggle"

# Files at the repo root that should stay at the root (not treated as adapter
# files to move).
ROOT_FILES_TO_KEEP = {"README.md", ".gitattributes"}

# ---------------------------------------------------------------------------


def reorganize_existing_adapter(api: HfApi, dry_run: bool) -> None:
    """Move files currently at the repo root into EXISTING_ADAPTER_SUBFOLDER."""
    all_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

    root_files = [
        f for f in all_files
        if "/" not in f and f not in ROOT_FILES_TO_KEEP
    ]

    if not root_files:
        print("No root-level adapter files found to move — skipping.")
        return

    print(f"Files to move into '{EXISTING_ADAPTER_SUBFOLDER}/':")
    for f in root_files:
        print(f"  {f} -> {EXISTING_ADAPTER_SUBFOLDER}/{f}")

    if dry_run:
        print("(dry run — no changes made)\n")
        return

    operations = []
    for f in root_files:
        dest = f"{EXISTING_ADAPTER_SUBFOLDER}/{f}"
        operations.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=dest))
        operations.append(CommitOperationDelete(path_in_repo=f))

    api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        operations=operations,
        commit_message=f"Move existing adapter into {EXISTING_ADAPTER_SUBFOLDER}/ subfolder",
    )
    print("Done reorganizing existing adapter.\n")


def download_blob_with_progress(blob: storage.Blob, dest_path: Path) -> None:
    """Download a blob to dest_path, showing a byte-level progress bar."""
    blob.reload()  # populate blob.size
    total = blob.size or 0

    with blob.open("rb") as src, open(dest_path, "wb") as dst, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path.name
    ) as pbar:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            pbar.update(len(chunk))


def list_adapter_folders(bucket: storage.Bucket, prefix: str) -> list[str]:
    """Return the top-level 'folder' names under prefix in the bucket."""
    iterator = bucket.list_blobs(prefix=prefix, delimiter="/")
    list(iterator)  # must consume to populate .prefixes
    folder_prefixes = iterator.prefixes  # e.g. {'adapters/run1/', 'adapters/run2/'}

    folders = []
    for p in sorted(folder_prefixes):
        name = p[len(prefix):].strip("/")
        if name:
            folders.append(name)
    return folders


def upload_new_adapters(api: HfApi, dry_run: bool) -> None:
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)

    folder_names = list_adapter_folders(bucket, GCS_PREFIX)

    if not folder_names:
        print(f"No adapter folders found under gs://{GCS_BUCKET_NAME}/{GCS_PREFIX}")
        return

    print(f"Found {len(folder_names)} adapter folder(s) in bucket:")
    for name in folder_names:
        print(f"  {name}")
    print()

    if dry_run:
        print("(dry run — would download each folder and upload to "
              f"{HF_REPO_ID}/<folder_name>/)\n")
        return

    common_subpath = GCS_COMMON_SUBPATH.strip("/")

    for name in folder_names:
        gcs_prefix = f"{GCS_PREFIX}{name}/"
        if common_subpath:
            gcs_prefix = f"{gcs_prefix}{common_subpath}/"
        print(f"Processing adapter '{name}' (gs://{GCS_BUCKET_NAME}/{gcs_prefix})")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            blobs = [b for b in bucket.list_blobs(prefix=gcs_prefix) if not b.name.endswith("/")]

            if not blobs:
                print("  no files found, skipping")
                continue

            for blob in blobs:
                rel_path = blob.name[len(gcs_prefix):]
                if not rel_path:
                    continue
                dest_file = tmp_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                download_blob_with_progress(blob, dest_file)
                print(f"  downloaded {rel_path}")

            api.upload_folder(
                folder_path=str(tmp_path),
                path_in_repo=name,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                commit_message=f"Add adapter '{name}'",
            )
            print(f"  uploaded to {HF_REPO_ID}/{name}/\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview actions without modifying the HF repo")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN_W")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN_W not found. Add HF_TOKEN_W=hf_... to .env in this directory."
        )

    api = HfApi(token=hf_token)

    print("=== Step 2: Upload new adapters from GCS ===")
    upload_new_adapters(api, args.dry_run)


if __name__ == "__main__":
    main()
