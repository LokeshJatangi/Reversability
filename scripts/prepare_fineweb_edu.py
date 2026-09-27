#!/usr/bin/env python3
"""Stream, tokenize, and freeze an exact-size FineWeb-Edu corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Iterable

import datasets
import numpy as np
import tokenizers
import transformers
from datasets import load_dataset
import huggingface_hub
from huggingface_hub import HfApi
from transformers import AutoTokenizer

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "v1.0.0"
TOKENIZER_ID = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/fineweb_edu_gpt2_50m_v1"))
    parser.add_argument("--train-targets", type=int, default=50_000_000)
    parser.add_argument("--validation-targets", type=int, default=1_000_000)
    parser.add_argument("--batch-documents", type=int, default=64)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def token_batches(rows: Iterable[dict], tokenizer, batch_documents: int) -> Iterable[list[int]]:
    texts: list[str] = []
    for row in rows:
        text = row.get("text")
        if not text:
            continue
        texts.append(text)
        if len(texts) == batch_documents:
            encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for ids in encoded:
                yield [*ids, tokenizer.eos_token_id]
            texts.clear()
    if texts:
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for ids in encoded:
            yield [*ids, tokenizer.eos_token_id]


def write_exact_splits(rows: Iterable[dict], tokenizer, output_dir: Path, train_targets: int,
                       validation_targets: int, batch_documents: int) -> dict:
    if train_targets < 1 or validation_targets < 1:
        raise ValueError("target counts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {"train": train_targets + 1, "validation": validation_targets + 1}
    arrays = {
        name: np.memmap(output_dir / f"{name}.bin", dtype=np.uint16, mode="w+", shape=(size,))
        for name, size in sizes.items()
    }
    split = "validation"
    offsets = {"train": 0, "validation": 0}
    documents = 0
    started = time.monotonic()
    for ids in token_batches(rows, tokenizer, batch_documents):
        documents += 1
        cursor = 0
        while cursor < len(ids):
            remaining = sizes[split] - offsets[split]
            take = min(remaining, len(ids) - cursor)
            arrays[split][offsets[split]:offsets[split] + take] = ids[cursor:cursor + take]
            offsets[split] += take
            cursor += take
            if offsets[split] == sizes[split]:
                arrays[split].flush()
                if split == "validation":
                    split = "train"
                else:
                    elapsed = time.monotonic() - started
                    return {"documents_consumed": documents, "elapsed_seconds": elapsed, "tokens": sizes}
        if documents % 1_000 == 0:
            total = offsets["validation"] + offsets["train"]
            print(f"documents={documents:,} tokens={total:,}/{sum(sizes.values()):,}", flush=True)
    raise RuntimeError(f"source exhausted at offsets {offsets}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    api = HfApi()
    dataset_sha = api.dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    tokenizer_sha = api.model_info(TOKENIZER_ID, revision=TOKENIZER_REVISION).sha
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=tokenizer_sha, use_fast=True)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("tokenizer vocabulary does not fit uint16")
    rows = load_dataset(
        DATASET_ID, name=DATASET_CONFIG, split="train", streaming=True, revision=dataset_sha
    )
    stats = write_exact_splits(
        rows, tokenizer, args.output_dir, args.train_targets, args.validation_targets,
        args.batch_documents,
    )
    tokenizer_dir = args.output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    files = {
        name: file_record(path)
        for name, path in {
            "train": args.output_dir / "train.bin",
            "validation": args.output_dir / "validation.bin",
        }.items()
    }
    tokenizer_files = {
        str(path.relative_to(tokenizer_dir)): file_record(path)
        for path in sorted(tokenizer_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "artifact_name": "data_fineweb_edu_gpt2_50m_v1",
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "requested_revision": DATASET_REVISION,
                    "resolved_revision": dataset_sha, "split": "train", "streaming": True},
        "tokenizer": {"id": TOKENIZER_ID, "requested_revision": TOKENIZER_REVISION,
                      "resolved_revision": tokenizer_sha, "vocab_size": tokenizer.vocab_size,
                      "eos_token_id": tokenizer.eos_token_id, "add_special_tokens": False,
                      "files": tokenizer_files},
        "split_policy": "first validation token stream, then training token stream; EOS after each document",
        "dtype": "uint16",
        "training_targets": args.train_targets,
        "validation_targets": args.validation_targets,
        "files": files,
        "source_documents_consumed": stats["documents_consumed"],
        "preparation_elapsed_seconds": stats["elapsed_seconds"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "datasets": datasets.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
        "command": shlex.join(sys.argv),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **stats, "files": files}, indent=2))


if __name__ == "__main__":
    main()
