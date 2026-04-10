# Datasets

This folder contains CSV datasets used for ADPr binding-site prediction and a sync script for managing them on HuggingFace Hub.

## Files

| File | Description |
|------|-------------|
| `adpr_sites_train.csv` | Training split (windowed sequences) |
| `adpr_sites_test.csv` | Test split (full protein sequences) |
| `hf_sync.py` | CLI tool for pushing/pulling datasets to/from HuggingFace Hub |

## hf_sync.py

A command-line tool to push local CSV datasets to [HuggingFace Hub](https://huggingface.co/) or pull remote datasets back down. Supports uploading as raw CSV files or converting to HuggingFace's Parquet format.

### Prerequisites

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Log in to HuggingFace (one-time setup):

```bash
huggingface-cli login
```

### Push a dataset

```bash
python datasets/hf_sync.py push <csv_file> [--format csv|parquet] [--repo REPO] [--private]
```

| Argument | Description |
|----------|-------------|
| `csv_file` | Path to the CSV file (relative to `datasets/` or absolute) |
| `--format` | `parquet` (default) converts to HF dataset format; `csv` uploads the raw file |
| `--repo` | Full repo id, e.g. `jbenbudd/my-dataset`. If omitted, derived from the filename |
| `--private` | Create the repo as private |

**Examples:**

```bash
# Push as Parquet (default) -- creates dataset card + Parquet shards
python datasets/hf_sync.py push adpr_sites_train.csv

# Push the raw CSV without conversion
python datasets/hf_sync.py push adpr_sites_train.csv --format csv

# Push to a specific repo name, privately
python datasets/hf_sync.py push adpr_sites_train.csv --repo jbenbudd/custom-name --private
```

When no `--repo` is given, the repo name is derived from the filename by replacing underscores with hyphens and dropping the extension. For example, `adpr_sites_train.csv` becomes `jbenbudd/adpr-sites-train`.

### Pull a dataset

```bash
python datasets/hf_sync.py pull <repo> [--output DIR]
```

| Argument | Description |
|----------|-------------|
| `repo` | HuggingFace dataset repo id (e.g. `jbenbudd/adpr-sites-train`). If no namespace is given, `jbenbudd/` is prepended |
| `--output` | Output directory (defaults to `datasets/`) |

**Examples:**

```bash
# Pull to the datasets/ folder
python datasets/hf_sync.py pull jbenbudd/adpr-sites-train

# Pull using just the dataset name (namespace defaults to jbenbudd)
python datasets/hf_sync.py pull adpr-sites-train

# Pull to a custom directory
python datasets/hf_sync.py pull jbenbudd/adpr-sites-train --output ./my-data
```

Pull first attempts to load the repo as a HuggingFace dataset (Parquet) and save each split as a CSV. If that fails, it falls back to downloading the raw repo contents via `snapshot_download`.
