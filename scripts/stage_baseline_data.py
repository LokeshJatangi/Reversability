#!/usr/bin/env python3
"""Locate an existing baseline export and stage its exact dataset in Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path

RUN_NAME = "baseline_20m_fineweb_edu_50m_v1"
DATA_NAME = "data_fineweb_edu_gpt2_50m_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_record(root: Path) -> dict:
    config = root / "configs/baseline_colab.json"
    metrics = root / "runs" / RUN_NAME / "metrics.jsonl"
    with metrics.open() as handle:
        startup = next(json.loads(line) for line in handle
                       if json.loads(line).get("event") == "startup")
    if sha(config) != startup["config_sha256"]:
        raise ValueError(f"Baseline config hash mismatch: {config}")
    return startup


def resolve_artifacts(root: Path, search_root: Path) -> tuple[Path, dict]:
    if (root / "configs/baseline_colab.json").is_file():
        return root, baseline_record(root)
    found = []
    for config in sorted(search_root.rglob("baseline_colab.json")):
        candidate = config.parent.parent
        if config.parent.name == "configs" and (candidate / "runs" / RUN_NAME / "metrics.jsonl").is_file():
            found.append(candidate)
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected baseline config at {root / 'configs/baseline_colab.json'}. "
            f"Found {len(found)} baseline artifact roots under {search_root}: {found}. "
            "Restore the baseline export inside MyDrive/Reversability, or pass its "
            "actual artifacts directory with --artifacts. Keep config, run logs and dataset together."
        )
    return found[0], baseline_record(found[0])


def validate_data(root: Path, expected_manifest_hash: str) -> None:
    manifest_path = root / "manifest.json"
    if sha(manifest_path) != expected_manifest_hash:
        raise ValueError(f"Not the baseline manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    records = [(root / f"{name}.bin", manifest["files"][name]) for name in ("train", "validation")]
    records += [(root / "tokenizer" / name, record)
                for name, record in manifest["tokenizer"]["files"].items()]
    for path, record in records:
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Baseline asset size/hash mismatch: {path}")


def stage_data(artifacts: Path, local: Path, search_root: Path) -> dict:
    artifacts, startup = resolve_artifacts(artifacts.resolve(), search_root.resolve())
    expected = startup["manifest_sha256"]
    config = json.loads((artifacts / "configs/baseline_colab.json").read_text())
    candidates = [artifacts / "data" / DATA_NAME, Path(config["data_dir"]), local]
    seen, rejected = set(), []
    source = None

    def accept(candidate: Path) -> bool:
        nonlocal source
        candidate = candidate.resolve()
        if candidate in seen:
            return False
        seen.add(candidate)
        try:
            validate_data(candidate, expected)
        except (OSError, ValueError, KeyError) as error:
            rejected.append(f"{candidate}: {error}")
            return False
        source = candidate
        return True

    for candidate in candidates:
        if accept(candidate):
            break
    if source is None:
        for manifest in sorted(search_root.rglob("manifest.json")):
            if accept(manifest.parent):
                break
    if source is None:
        raise FileNotFoundError(
            f"No dataset matching baseline manifest SHA-256 {expected} was found. "
            f"Expected folder: {artifacts / 'data' / DATA_NAME}. "
            f"Searched known paths and {search_root}. Restore the original exported "
            "dataset folder (manifest.json, train.bin, validation.bin, tokenizer/) "
            "inside MyDrive/Reversability and rerun. Preparing a new dataset manifest "
            "does not preserve the recorded baseline comparison.\n" + "\n".join(rejected)
        )
    local = local.resolve()
    if source != local:
        local.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=local.name + ".staging-", dir=local.parent))
        shutil.copytree(source, stage, dirs_exist_ok=True)
        validate_data(stage, expected)
        if local.exists():
            backup = local.with_name(local.name + ".previous-" + uuid.uuid4().hex)
            local.rename(backup)
            print(f"Preserved previous runtime data at {backup}", flush=True)
        stage.rename(local)
    return {"artifacts": str(artifacts), "source_data": str(source),
            "local_data": str(local), "manifest_sha256": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = stage_data(args.artifacts, args.local_data, args.search_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
