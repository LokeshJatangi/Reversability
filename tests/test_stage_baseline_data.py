import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage_baseline_data import DATA_NAME, RUN_NAME, stage_data


def fixture(root):
    data = root / "data" / DATA_NAME
    data.mkdir(parents=True)
    (data / "tokenizer").mkdir()
    for name in ["train.bin", "validation.bin", "tokenizer/tokenizer.json"]:
        (data / name).write_bytes(b"frozen fixture")
    def record(path):
        return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"files": {name: record(data / f"{name}.bin") for name in ["train", "validation"]},
                "tokenizer": {"files": {"tokenizer.json": record(data / "tokenizer/tokenizer.json")}}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    config = root / "configs/baseline_colab.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"data_dir": "/nonexistent/old-colab-runtime"}))
    metrics = root / "runs" / RUN_NAME / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"event": "startup", "config_sha256": record(config)["sha256"],
                                   "manifest_sha256": record(data / "manifest.json")["sha256"]}) + "\n")
    return data


def test_nested_export_is_discovered_and_verified(tmp_path):
    drive = tmp_path / "Reversability"
    actual = drive / "downloaded-export/artifacts"
    data = fixture(actual)
    local = tmp_path / "runtime/data"
    result = stage_data(drive / "artifacts", local, drive)
    assert result["artifacts"] == str(actual)
    assert result["source_data"] == str(data)
    assert (local / "manifest.json").read_bytes() == (data / "manifest.json").read_bytes()


def test_moved_data_and_existing_runtime_are_preserved(tmp_path):
    root = tmp_path / "artifacts"
    data = fixture(root)
    moved = tmp_path / "restored-data"
    data.rename(moved)
    local = tmp_path / "runtime"
    local.mkdir()
    (local / "user-file").write_text("preserve")
    result = stage_data(root, local, tmp_path)
    assert result["source_data"] == str(moved)
    backup = next(tmp_path.glob("runtime.previous-*"))
    assert (backup / "user-file").read_text() == "preserve"
    assert (local / "train.bin").is_file()


def test_corrupt_data_is_rejected_without_creating_runtime(tmp_path):
    root = tmp_path / "artifacts"
    data = fixture(root)
    (data / "train.bin").write_bytes(b"wrong tokens")
    local = tmp_path / "runtime"
    with pytest.raises(FileNotFoundError, match="No dataset matching baseline"):
        stage_data(root, local, tmp_path)
    assert not local.exists()


def test_different_manifest_with_same_tokens_is_rejected(tmp_path):
    root = tmp_path / "artifacts"
    data = fixture(root)
    with (data / "manifest.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(FileNotFoundError, match="Not the baseline manifest"):
        stage_data(root, tmp_path / "runtime", tmp_path)


@pytest.mark.parametrize("filename", ["baseline-inputs-recovery.zip", "baseline-reference-recovery.zip"])
def test_recovery_zip_restores_into_new_folder(tmp_path, filename):
    export = tmp_path / "laptop/artifacts"
    fixture(export)
    drive = tmp_path / "MyDrive/Reversability"
    drive.mkdir(parents=True)
    with zipfile.ZipFile(drive / filename, "w") as archive:
        for path in export.rglob("*"):
            if path.is_file():
                archive.write(path, Path("artifacts") / path.relative_to(export))
    result = stage_data(drive / "artifacts", tmp_path / "runtime", drive.parent)
    recovered = Path(result["artifacts"])
    assert recovered.is_relative_to(drive)
    assert recovered != export
    assert (recovered / "data" / DATA_NAME / "train.bin").is_file()
    assert (tmp_path / "runtime/manifest.json").is_file()
    again = stage_data(drive / "artifacts", tmp_path / "runtime", drive.parent)
    assert again == result
    assert len(list(drive.glob("baseline-recovery-*"))) == 1
    assert not list(tmp_path.glob("runtime.previous-*"))


def test_recovery_zip_rejects_path_escape(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    with zipfile.ZipFile(drive / "baseline-reference-recovery.zip", "w") as archive:
        archive.writestr("../escaped-file", "bad")
    with pytest.raises(ValueError, match="Invalid recovery archive entry"):
        stage_data(drive / "artifacts", tmp_path / "runtime", drive)
    assert not (drive / "escaped-file").exists()
