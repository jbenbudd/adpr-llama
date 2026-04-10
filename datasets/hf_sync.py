#!/usr/bin/env python3
"""Push local CSV datasets to HuggingFace Hub or pull remote datasets down."""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_NAMESPACE = "jbenbudd"
SCRIPT_DIR = Path(__file__).resolve().parent


def repo_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return f"{DEFAULT_NAMESPACE}/{stem.replace('_', '-')}"


def resolve_csv_path(file_arg: str) -> Path:
    """Resolve a CSV path relative to the datasets/ directory or as-is."""
    p = Path(file_arg)
    if p.is_absolute() and p.exists():
        return p
    relative = SCRIPT_DIR / p
    if relative.exists():
        return relative
    if p.exists():
        return p.resolve()
    print(f"Error: file not found: {file_arg}", file=sys.stderr)
    sys.exit(1)


def push_csv(csv_path: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=str(csv_path),
        path_in_repo=csv_path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {csv_path.name} to https://huggingface.co/datasets/{repo_id}")


def push_parquet(csv_path: Path, repo_id: str, private: bool) -> None:
    from datasets import Dataset

    ds = Dataset.from_csv(str(csv_path))
    ds.push_to_hub(repo_id, private=private)
    print(
        f"Pushed {csv_path.name} (as Parquet) to "
        f"https://huggingface.co/datasets/{repo_id}"
    )


def push(args: argparse.Namespace) -> None:
    csv_path = resolve_csv_path(args.file)
    repo_id = args.repo or repo_name_from_filename(csv_path.name)

    if args.format == "csv":
        push_csv(csv_path, repo_id, private=args.private)
    else:
        push_parquet(csv_path, repo_id, private=args.private)


def pull(args: argparse.Namespace) -> None:
    repo_id = args.repo
    if "/" not in repo_id:
        repo_id = f"{DEFAULT_NAMESPACE}/{repo_id}"

    output_dir = Path(args.output) if args.output else SCRIPT_DIR

    try:
        _pull_hf_dataset(repo_id, output_dir)
    except Exception:
        _pull_snapshot(repo_id, output_dir)


def _pull_hf_dataset(repo_id: str, output_dir: Path) -> None:
    from datasets import load_dataset

    ds = load_dataset(repo_id)
    for split_name, split_ds in ds.items():
        dest = output_dir / f"{repo_id.split('/')[-1]}_{split_name}.csv"
        split_ds.to_csv(str(dest), index=False)
        print(f"Saved {split_name} split -> {dest}")


def _pull_snapshot(repo_id: str, output_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(output_dir / repo_id.split("/")[-1]),
    )
    print(f"Downloaded {repo_id} -> {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync datasets with HuggingFace Hub."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- push --
    push_parser = subparsers.add_parser("push", help="Push a local CSV to HuggingFace")
    push_parser.add_argument(
        "file", help="CSV file to push (relative to datasets/ or absolute path)"
    )
    push_parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Upload format: raw csv or converted parquet (default: parquet)",
    )
    push_parser.add_argument(
        "--repo",
        default=None,
        help=(
            f"Full repo id (e.g. {DEFAULT_NAMESPACE}/my-dataset). "
            "Defaults to namespace/filename-derived-name."
        ),
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private",
    )
    push_parser.set_defaults(func=push)

    # -- pull --
    pull_parser = subparsers.add_parser(
        "pull", help="Pull a dataset from HuggingFace"
    )
    pull_parser.add_argument(
        "repo",
        help=(
            "HuggingFace dataset repo id (e.g. jbenbudd/adpr-sites-train). "
            "If no namespace is given, defaults to "
            f"'{DEFAULT_NAMESPACE}/<name>'."
        ),
    )
    pull_parser.add_argument(
        "--output",
        default=None,
        help="Output directory (defaults to the datasets/ folder)",
    )
    pull_parser.set_defaults(func=pull)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
