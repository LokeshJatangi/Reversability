import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversibility import BaselineLM, ModelConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_baseline
from train_baseline import batch_at
from prepare_fineweb_edu import file_record, write_exact_splits


def test_model_shape_tied_weights_and_gradients():
    model = BaselineLM(ModelConfig(vocab_size=101, max_sequence_length=16, n_layers=2,
                                   n_heads=2, d_model=32, d_ff=64))
    inputs = torch.randint(0, 101, (2, 16))
    logits = model(inputs)
    assert logits.shape == (2, 16, 101)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), inputs[:, 1:].flatten()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_partial_batch_has_exact_target_count(tmp_path):
    path = tmp_path / "tokens.bin"
    data = np.memmap(path, dtype=np.uint16, mode="w+", shape=(12,))
    data[:] = np.arange(12); data.flush()
    x, y, count = batch_at(data, cursor=0, batch_size=2, sequence_length=8, target_limit=11)
    assert count == 11
    assert x.shape == y.shape == (2, 8)
    assert int((y != -100).sum()) == 11
    assert y.flatten()[:11].tolist() == list(range(1, 12))


def test_full_configuration_is_near_20m_parameters():
    model = BaselineLM(ModelConfig())
    assert 19_000_000 <= model.parameter_count() <= 21_000_000


class TinyTokenizer:
    eos_token_id = 99

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[ord(character) - 96 for character in text] for text in texts]}


def test_preparation_writes_exact_nonoverlapping_splits(tmp_path):
    rows = iter([{"text": "abc"}, {"text": "de"}, {"text": "fghij"}])
    stats = write_exact_splits(rows, TinyTokenizer(), tmp_path, train_targets=5,
                               validation_targets=4, batch_documents=2)
    validation = np.memmap(tmp_path / "validation.bin", dtype=np.uint16, mode="r")
    train = np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r")
    assert validation.tolist() == [1, 2, 3, 99, 4]
    assert train.tolist() == [5, 99, 6, 7, 8, 9]
    assert stats["tokens"] == {"train": 6, "validation": 5}


def test_file_record_hashes_tokenizer_assets(tmp_path):
    asset = tmp_path / "tokenizer.json"
    asset.write_bytes(b"frozen-tokenizer")
    assert file_record(asset) == {
        "bytes": 16,
        "sha256": hashlib.sha256(b"frozen-tokenizer").hexdigest(),
    }


def _write_tiny_training_fixture(tmp_path: Path, output_name: str) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    for name, targets in (("train", 32), ("validation", 8)):
        values = np.arange(targets + 1, dtype=np.uint16) % 31
        path = data_dir / f"{name}.bin"
        values.tofile(path)
    manifest = {
        "training_targets": 32,
        "validation_targets": 8,
        "files": {
            name: file_record(data_dir / f"{name}.bin")
            for name in ("train", "validation")
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "experiment_name": "resume_test",
        "data_dir": str(data_dir),
        "output_dir": str(tmp_path / output_name),
        "seed": 123,
        "sequence_length": 4,
        "physical_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_train_targets": 32,
        "eval_targets": 8,
        "eval_interval": 999,
        "checkpoint_interval": 1,
        "learning_rate": 0.001,
        "min_learning_rate": 0.0001,
        "warmup_targets": 16,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "precision": "bf16",
        "compile": False,
        "model": {"vocab_size": 31, "max_sequence_length": 4, "n_layers": 1,
                  "n_heads": 1, "d_model": 8, "d_ff": 16, "dropout": 0.0,
                  "bias": False},
    }
    path = tmp_path / f"{output_name}.json"
    path.write_text(json.dumps(config))
    return path


def _run_training(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["train_baseline.py", *map(str, arguments)])
    train_baseline.main()


def test_resume_matches_uninterrupted_optimizer_boundary(tmp_path, monkeypatch):
    uninterrupted_config = _write_tiny_training_fixture(tmp_path, "uninterrupted")
    resumed_config = _write_tiny_training_fixture(tmp_path, "resumed")
    _run_training(monkeypatch, ["--config", uninterrupted_config, "--device", "cpu"])
    _run_training(monkeypatch, ["--config", resumed_config, "--device", "cpu",
                                "--stop-after-targets", "16"])
    resume_path = tmp_path / "resumed" / "latest.pt"
    _run_training(monkeypatch, ["--config", resumed_config, "--device", "cpu",
                                "--resume", resume_path])
    uninterrupted = torch.load(tmp_path / "uninterrupted" / "latest.pt", weights_only=False)
    resumed = torch.load(resume_path, weights_only=False)
    assert uninterrupted["committed_targets"] == resumed["committed_targets"] == 32
    assert uninterrupted["step"] == resumed["step"] == 2
    for name, expected in uninterrupted["model"].items():
        torch.testing.assert_close(resumed["model"][name], expected, rtol=0, atol=0)
    for parameter_id, expected_state in uninterrupted["optimizer"]["state"].items():
        actual_state = resumed["optimizer"]["state"][parameter_id]
        for name, expected in expected_state.items():
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
            else:
                assert actual_state[name] == expected
