#!/usr/bin/env python3
"""Package the verified baseline reference for one-time Drive recovery."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from stage_baseline_data import DATA_NAME, RUN_NAME, baseline_record, sha, validate_data


def bundle(root: Path, output: Path, without_checkpoints: bool = False) -> dict:
    startup = baseline_record(root)
    validate_data(root / "data" / DATA_NAME, startup["manifest_sha256"])
    run = root / "runs" / RUN_NAME
    summary = json.loads((run / "run_summary.json").read_text())
    if summary["committed_targets"] != 50_000_000 or not summary["completed_full_target_budget"]:
        raise ValueError("Baseline reference must be a completed 50M-target run")
    if not without_checkpoints:
        for name in ["latest.pt", "best.pt"]:
            if not (run / name).is_file():
                raise FileNotFoundError(run / name)
    if output.exists():
        raise FileExistsError(output)
    # Keep all metadata, source, data and full-run checkpoints. The separate
    # smoke checkpoints are not needed to validate or compare the full baseline.
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and not (without_checkpoints and p.suffix == ".pt")
             and not ("_smoke" in str(p.relative_to(root)) and p.suffix == ".pt")]
    records = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, Path("artifacts") / path.relative_to(root))
        archive.writestr("recovery-manifest.json", json.dumps(records, indent=2) + "\n")
    # Read and hash the archive contents, not just the input files.
    import hashlib
    with zipfile.ZipFile(output) as archive:
        for name, record in records.items():
            digest = hashlib.sha256()
            with archive.open("artifacts/" + name) as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Recovery archive verification failed: {name}")
    return {"archive": str(output.resolve()), "files": len(files), "bytes": output.stat().st_size,
            "sha256": sha(output), "excluded": "all .pt checkpoints; originals preserved" if without_checkpoints
            else "separate smoke .pt checkpoints; originals preserved"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-checkpoints", action="store_true",
                        help="Small comparison-input recovery; keep checkpoint evidence locally")
    args = parser.parse_args()
    print(json.dumps(bundle(args.artifacts, args.output, args.without_checkpoints), indent=2), flush=True)


if __name__ == "__main__":
    main()
