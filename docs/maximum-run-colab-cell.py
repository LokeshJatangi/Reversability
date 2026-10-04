"""Run after the existing reversible notebook setup/data/review cells.

This handoff installs the audited planning decision, not new training code.
Existing Drive data, checkpoints and frozen execution source must be preserved.
"""
import hashlib
import json
import os
from pathlib import Path

study = Path(ARTIFACTS) / "reversible-v1"
decision_bytes = (Path(REPO) / "docs/review-decision-2026-10-04.json").read_bytes()
decision = json.loads(decision_bytes)
for relative, key in (
    ("review-proposal.json", "review_proposal_sha256"),
    ("source-manifest.json", "source_manifest_sha256"),
    ("correctness/cuda.json", "cuda_gate_sha256"),
):
    actual = hashlib.sha256((study / relative).read_bytes()).hexdigest()
    if actual != decision["evidence"][key]:
        raise RuntimeError(f"Audited evidence differs: {relative}; preserve files and request review")
proposal = json.loads((study / "review-proposal.json").read_text())
if decision["selected_method"] != proposal["selected_method"] or decision["policy"] != proposal["policy"]:
    raise RuntimeError("Decision does not match the frozen proposal")
destination = study / "review-decision.json"
if destination.exists():
    if json.loads(destination.read_text()) != decision:
        raise RuntimeError("A different review decision already exists; it was not overwritten")
else:
    # Flush the decision before launching the existing checkpointed runner.
    temporary = study / "review-decision.json.pending"
    with temporary.open("xb") as handle:
        handle.write(decision_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
print("Planning review saved:", destination)
print("Starting midpoint maximum-batch search/run; completed matched runs are preserved.")
run_logged(
    [sys.executable, "-u", Path(REPO) / "scripts/run_reversible_study.py",
     "maximum", "--artifacts", ARTIFACTS, "--max-batch", "256"],
    study / "maximum-orchestration.log",
)
