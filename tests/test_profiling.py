import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_training as profiling
from test_baseline import _write_tiny_training_fixture


@pytest.mark.parametrize("method", ["baseline", "midpoint"])
@pytest.mark.parametrize("mode", ["trace", "memory"])
def test_annotations_preserve_complete_optimizer_update(tmp_path, method, mode):
    config = json.loads(_write_tiny_training_fixture(tmp_path, "fixture").read_text())
    config.update(method=method, backward_mode="reconstruct", precision="fp32")
    train = np.memmap(Path(config["data_dir"]) / "train.bin", mode="r", dtype=np.uint16)
    def run(instrumented):
        torch.manual_seed(91)
        model = profiling.build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .95))
        regions = profiling.Regions(torch.device("cpu"), mode if instrumented else "timing")
        original_pair = profiling.reversible.forward_pair
        original_grad = torch.autograd.grad
        if instrumented:
            with profiling.annotations(model, regions):
                result = profiling.optimizer_update(model, optimizer, None, train, config, 0,
                                                     torch.device("cpu"), regions)
        else:
            result = profiling.optimizer_update(model, optimizer, None, train, config, 0,
                                                 torch.device("cpu"), regions)
        assert original_pair is profiling.reversible.forward_pair
        assert original_grad is torch.autograd.grad
        assert "forward" not in model.lm_head.__dict__
        profiling.check_update(result)
        return model, optimizer, result
    plain_model, plain_optimizer, plain = run(False)
    marked_model, marked_optimizer, marked = run(True)
    assert plain["loss"] == marked["loss"]
    assert plain["targets"] == marked["targets"] == 16
    for name, expected in plain_model.state_dict().items():
        torch.testing.assert_close(marked_model.state_dict()[name], expected, rtol=0, atol=0)
    for pid, state in plain_optimizer.state_dict()["state"].items():
        for name, value in state.items():
            actual = marked_optimizer.state_dict()["state"][pid][name]
            torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_annotations_restore_after_failure(tmp_path):
    config = json.loads(_write_tiny_training_fixture(tmp_path, "fixture").read_text())
    config.update(method="midpoint")
    model = profiling.build_model(config)
    pair, grad = profiling.reversible.inverse_pair, torch.autograd.grad
    with pytest.raises(RuntimeError, match="interrupted"):
        with profiling.annotations(model, profiling.Regions(torch.device("cpu"), "trace")):
            raise RuntimeError("interrupted")
    assert pair is profiling.reversible.inverse_pair
    assert grad is torch.autograd.grad
    assert "forward" not in model.blocks[0].attn.__dict__


def test_worker_rejects_changed_data_and_preserves_failure(tmp_path):
    from argparse import Namespace
    path = _write_tiny_training_fixture(tmp_path, "fixture")
    config = json.loads(path.read_text())
    data = Path(config["data_dir"])
    (data / "train.bin").write_bytes(b"wrong data")
    output = tmp_path / "profile"
    args = Namespace(config=path, data_dir=data, output=output, method="baseline", batch=2,
                     mode="timing", cpu_smoke=True, warmup=1, updates=1, trace_updates=1)
    with pytest.raises(ValueError, match="size/hash mismatch"):
        profiling.worker(args)
    assert json.loads((output / "result.json").read_text())["status"] == "failed"


def test_worker_refuses_existing_output(tmp_path):
    from argparse import Namespace
    path = _write_tiny_training_fixture(tmp_path, "fixture")
    output = tmp_path / "profile"
    output.mkdir()
    sentinel = output / "result.json"
    sentinel.write_text("preserved")
    args = Namespace(config=path, data_dir=Path(json.loads(path.read_text())["data_dir"]),
                     output=output, method="baseline", batch=2, mode="timing", cpu_smoke=True)
    with pytest.raises(FileExistsError):
        profiling.worker(args)
    assert sentinel.read_text() == "preserved"


def test_cli_resolves_paths_before_worker_changes_directory(tmp_path, monkeypatch):
    path = _write_tiny_training_fixture(tmp_path, "fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["profile_training.py", "--config", path.name,
        "--data-dir", "data", "--output", "new-profile", "--cpu-smoke"])
    args = profiling.parse_args()
    assert args.config == path.resolve()
    assert args.data_dir == (tmp_path / "data").resolve()
    assert args.output == (tmp_path / "new-profile").resolve()
