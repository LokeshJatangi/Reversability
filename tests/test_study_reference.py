import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_reversible_study import completed


def reference(run):
    summary = {"completed_full_target_budget": True, "committed_targets": 50_000_000,
               "target_budget": 50_000_000, "optimizer_steps": 1684,
               "final_validation_loss": 5.462294847167969}
    (run / "run_summary.json").write_text(json.dumps(summary))
    events = [{"event": "validation", "committed_targets": 50_000_000,
               "validation_loss": summary["final_validation_loss"]},
              {"event": "run_summary", **summary}]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return summary


def test_baseline_reference_does_not_require_trained_weights(tmp_path):
    summary = reference(tmp_path)
    assert completed(tmp_path, require_checkpoint=False) == summary
    assert not (tmp_path / "latest.pt").exists()


def test_baseline_reference_rejects_changed_summary(tmp_path):
    summary = reference(tmp_path)
    summary["final_validation_loss"] = 4.0
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="metrics/summary disagreement"):
        completed(tmp_path, require_checkpoint=False)


def test_new_reversible_run_still_requires_checkpoint(tmp_path):
    reference(tmp_path)
    with pytest.raises(FileNotFoundError):
        completed(tmp_path)
