#!/usr/bin/env python3
"""Train the ordinary-autograd baseline with exact committed-target accounting."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import random
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversibility import BaselineLM, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-targets", type=int, help="override for smoke/partial runs")
    parser.add_argument(
        "--stop-after-targets",
        type=int,
        help="stop at an optimizer boundary without changing the scheduled target budget",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_at(data: np.memmap, cursor: int, batch_size: int, sequence_length: int,
             target_limit: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    count = min(batch_size * sequence_length, target_limit - cursor)
    if count <= 0:
        raise StopIteration
    sequences = math.ceil(count / sequence_length)
    needed = sequences * sequence_length
    raw = np.asarray(data[cursor:cursor + count + 1], dtype=np.int64)
    x = np.zeros(needed, dtype=np.int64)
    y = np.full(needed, -100, dtype=np.int64)
    x[:count] = raw[:-1]
    y[:count] = raw[1:]
    return torch.from_numpy(x.reshape(sequences, sequence_length)), torch.from_numpy(
        y.reshape(sequences, sequence_length)), count


@torch.no_grad()
def evaluate(model: BaselineLM, data: np.memmap, sequence_length: int, targets: int,
             device: torch.device, amp_context) -> float:
    model.eval()
    total_loss = 0.0
    cursor = 0
    while cursor < targets:
        x, y, count = batch_at(data, cursor, 8, sequence_length, targets)
        x, y = x.to(device), y.to(device)
        with amp_context():
            logits = model(x)
            loss_sum = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
        total_loss += loss_sum.float().item()
        cursor += count
    model.train()
    return total_loss / targets


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def emit(event_path: Path, event_type: str, payload: dict) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }
    line = json.dumps(record)
    print(line, flush=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def runtime_metadata(device: torch.device) -> dict:
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["accelerator"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "device_count": torch.cuda.device_count(),
        }
    else:
        metadata["accelerator"] = None
    return metadata


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    data_dir = Path(config["data_dir"])
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path, validation_path = data_dir / "train.bin", data_dir / "validation.bin"
    for name, path in (("train", train_path), ("validation", validation_path)):
        if file_sha256(path) != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"dataset hash mismatch: {path}")
    max_targets = int(config["max_train_targets"]) if args.max_targets is None else args.max_targets
    if max_targets < 1:
        raise ValueError("target budget must be positive")
    if max_targets > manifest["training_targets"]:
        raise ValueError("requested targets exceed frozen training split")
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model_config = ModelConfig(**config["model"])
    model = BaselineLM(model_config).to(device)
    parameter_count = model.parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  betas=(config["beta1"], config["beta2"]),
                                  weight_decay=config["weight_decay"])
    precision = config["precision"]
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU does not support BF16; use a frozen FP16 configuration")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and precision == "fp16" else None
    train = np.memmap(train_path, dtype=np.uint16, mode="r")
    validation = np.memmap(validation_path, dtype=np.uint16, mode="r")
    output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / "metrics.jsonl"
    sequence_length = int(config["sequence_length"])
    physical_batch = int(config["physical_batch_size"])
    accumulation = int(config["gradient_accumulation_steps"])
    if sequence_length > model_config.max_sequence_length:
        raise ValueError("sequence_length exceeds the model maximum")
    if physical_batch < 1 or accumulation < 1:
        raise ValueError("batch size and accumulation must be positive")
    bytes_per_token = np.dtype(np.uint16).itemsize
    expected_sizes = {
        "train": (int(manifest["training_targets"]) + 1) * bytes_per_token,
        "validation": (int(manifest["validation_targets"]) + 1) * bytes_per_token,
    }
    for name, path in (("train", train_path), ("validation", validation_path)):
        if path.stat().st_size != expected_sizes[name]:
            raise RuntimeError(f"dataset size/count mismatch: {path}")
    committed, step = 0, 0
    last_validation_loss = None
    best_validation_loss = None
    best_validation_step = None
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    manifest_hash = file_sha256(manifest_path)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_hash or checkpoint["manifest_sha256"] != manifest_hash:
            raise RuntimeError("checkpoint configuration or dataset manifest mismatch")
        if checkpoint.get("target_budget") != max_targets:
            raise RuntimeError("checkpoint target budget mismatch")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        committed, step = checkpoint["committed_targets"], checkpoint["step"]
        last_validation_loss = checkpoint.get("last_validation_loss")
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_validation_step = checkpoint.get("best_validation_step")
        random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if scaler is not None:
            if checkpoint.get("grad_scaler") is None:
                raise RuntimeError("FP16 checkpoint is missing gradient-scaler state")
            scaler.load_state_dict(checkpoint["grad_scaler"])
    process_start_committed = committed
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_context = contextlib.nullcontext
    if config.get("compile"):
        model = torch.compile(model)
    update_capacity = physical_batch * sequence_length * accumulation
    run_limit = max_targets if args.stop_after_targets is None else args.stop_after_targets
    if not committed < run_limit <= max_targets:
        raise ValueError("stop-after target must be above the resumed cursor and at most the target budget")
    if run_limit != max_targets and run_limit % update_capacity:
        raise ValueError("stop-after target must be an ordinary optimizer-update boundary")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_sync(device)
    started = time.monotonic(); model.train()
    startup = {
        "experiment_name": config["experiment_name"],
        "parameters": parameter_count,
        "architecture": config["model"],
        "device": str(device),
        "precision": precision if device.type == "cuda" else "fp32",
        "physical_batch_size_sequences_per_device_microstep": physical_batch,
        "gradient_accumulation_steps": accumulation,
        "data_parallel_world_size": 1,
        "effective_batch_size_sequences_per_optimizer_update": physical_batch * accumulation,
        "regular_valid_targets_per_microstep": physical_batch * sequence_length,
        "regular_valid_targets_per_optimizer_update": update_capacity,
        "target_budget": max_targets,
        "run_stop_target": run_limit,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "command": shlex.join(sys.argv),
        "environment": runtime_metadata(device),
    }
    startup["resumed_from_targets"] = process_start_committed
    emit(event_path, "startup", startup)
    while committed < run_limit:
        window_python_rng = random.getstate()
        window_numpy_rng = np.random.get_state()
        window_torch_rng = torch.get_rng_state()
        window_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        optimizer.zero_grad(set_to_none=True)
        update_target_cap = min(run_limit, committed + update_capacity)
        update_targets = update_target_cap - committed
        processed = 0
        update_loss_sum = 0.0
        for _ in range(accumulation):
            if processed >= update_targets: break
            x, y, count = batch_at(train, committed + processed, physical_batch, sequence_length, update_target_cap)
            x, y = x.to(device), y.to(device)
            with amp_context():
                logits = model(x)
                loss_sum = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
                normalized_loss = loss_sum / update_targets
                if scaler is None:
                    normalized_loss.backward()
                else:
                    scaler.scale(normalized_loss).backward()
            update_loss_sum += loss_sum.detach().float().item()
            processed += count
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        progress = (committed + update_targets) / max_targets
        warmup = min(1.0, (committed + update_targets) / config["warmup_targets"])
        cosine = config["min_learning_rate"] + 0.5 * (config["learning_rate"] - config["min_learning_rate"]) * (1 + math.cos(math.pi * progress))
        learning_rate = min(config["learning_rate"] * warmup, cosine)
        for group in optimizer.param_groups: group["lr"] = learning_rate
        skipped = False
        if scaler is None:
            optimizer.step()
        else:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
        if skipped:
            random.setstate(window_python_rng)
            np.random.set_state(window_numpy_rng)
            torch.set_rng_state(window_torch_rng)
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(window_cuda_rng)
            emit(event_path, "mixed_precision_overflow", {
                "step": step,
                "committed_targets": committed,
                "retried_valid_targets": update_targets,
                "scale_before": scale_before,
                "scale_after": scaler.get_scale(),
            })
            continue
        committed += update_targets; step += 1
        if step % 10 == 0 or committed == run_limit:
            cuda_sync(device)
            elapsed = time.monotonic() - started
            emit(event_path, "training_progress", {
                "step": step,
                "committed_targets": committed,
                "valid_targets_this_update": update_targets,
                "training_loss_this_update": update_loss_sum / update_targets,
                "lr": learning_rate,
                "end_to_end_targets_per_second_this_process":
                    (committed - process_start_committed) / elapsed,
            })
        improved_validation = False
        if step % config["eval_interval"] == 0 or committed == max_targets:
            val_loss = evaluate(model, validation, sequence_length,
                                min(config["eval_targets"], manifest["validation_targets"]), device, amp_context)
            last_validation_loss = val_loss
            if best_validation_loss is None or val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_validation_step = step
                improved_validation = True
            emit(event_path, "validation", {
                "step": step,
                "committed_targets": committed,
                "validation_targets": min(config["eval_targets"], manifest["validation_targets"]),
                "validation_loss": val_loss,
                "best_validation_loss": best_validation_loss,
                "best_validation_step": best_validation_step,
            })
        checkpoint_due = step % config["checkpoint_interval"] == 0 or committed == run_limit
        if checkpoint_due:
            state = {"experiment_name": config["experiment_name"], "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                     "committed_targets": committed, "target_budget": max_targets,
                     "config_sha256": config_hash,
                     "manifest_sha256": manifest_hash, "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "grad_scaler": scaler.state_dict() if scaler is not None else None,
                     "last_validation_loss": last_validation_loss,
                     "best_validation_loss": best_validation_loss,
                     "best_validation_step": best_validation_step}
            atomic_save(state, output_dir / "latest.pt")
            emit(event_path, "checkpoint", {
                "step": step,
                "committed_targets": committed,
                "path": str(output_dir / "latest.pt"),
            })
            if improved_validation:
                atomic_save(state, output_dir / "best.pt")
                emit(event_path, "best_checkpoint", {
                    "step": step,
                    "committed_targets": committed,
                    "validation_loss": best_validation_loss,
                    "path": str(output_dir / "best.pt"),
                })
    assert committed == run_limit
    cuda_sync(device)
    elapsed = time.monotonic() - started
    summary = {
        "experiment_name": config["experiment_name"],
        "completed_full_target_budget": committed == max_targets,
        "committed_targets": committed,
        "target_budget": max_targets,
        "optimizer_steps": step,
        "final_validation_loss": last_validation_loss,
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
        "elapsed_seconds_this_process": elapsed,
        "end_to_end_targets_per_second_this_process": (committed - process_start_committed) / elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "checkpoint": str(output_dir / "latest.pt"),
        "best_checkpoint": str(output_dir / "best.pt") if (output_dir / "best.pt").exists() else None,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(event_path, "run_summary", summary)


if __name__ == "__main__":
    main()
