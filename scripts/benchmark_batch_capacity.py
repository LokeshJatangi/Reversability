#!/usr/bin/env python3
"""Find the largest repeatedly successful physical batch on one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversibility import BaselineLM, ModelConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/baseline_batch_capacity.json"))
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--headroom-fraction", type=float, default=0.10)
    parser.add_argument("--candidate", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def candidate_run(args: argparse.Namespace, config: dict) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the batch-capacity search")
    if not 0 < args.headroom_fraction < 1:
        raise ValueError("headroom fraction must be between zero and one")
    device = torch.device("cuda")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_config = ModelConfig(**config["model"])
    precision = config["precision"]
    if precision != "fp16":
        raise ValueError("the Colab capacity protocol currently requires frozen FP16")
    started = time.monotonic()
    payload = {
        "physical_batch_size": args.candidate,
        "repetitions": args.repetitions,
        "headroom_fraction": args.headroom_fraction,
        "status": "error",
    }
    try:
        model = build_model(config).to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda")
        sequence_length = int(config["sequence_length"])
        generator = torch.Generator(device=device).manual_seed(seed + args.candidate)
        inputs = torch.randint(
            0, model_config.vocab_size, (args.candidate, sequence_length),
            device=device, generator=generator,
        )
        targets = torch.randint(
            0, model_config.vocab_size, (args.candidate, sequence_length),
            device=device, generator=generator,
        )

        accumulation = int(config["gradient_accumulation_steps"])
        def update() -> float:
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(accumulation):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                scaler.scale(loss / accumulation).backward()
                loss_value += float(loss.detach()) / accumulation
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise FloatingPointError("mixed-precision overflow during capacity probe")
            if not math.isfinite(loss_value) or not torch.isfinite(norm):
                raise FloatingPointError("nonfinite loss or gradient during capacity probe")
            torch.cuda.synchronize(device)
            return loss_value

        torch.cuda.reset_peak_memory_stats(device)
        startup_loss = update()  # initializes Adam states and includes first-update peaks
        startup_allocated = torch.cuda.max_memory_allocated(device)
        startup_reserved = torch.cuda.max_memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        losses = [update() for _ in range(args.repetitions)]
        steady_allocated = torch.cuda.max_memory_allocated(device)
        steady_reserved = torch.cuda.max_memory_reserved(device)
        properties = torch.cuda.get_device_properties(device)
        peak_reserved = max(startup_reserved, steady_reserved)
        limit = math.floor(properties.total_memory * (1 - args.headroom_fraction))
        payload.update({
            "status": "pass" if peak_reserved <= limit else "insufficient_headroom",
            "valid_targets_per_microstep": args.candidate * sequence_length,
            "gradient_accumulation_steps": accumulation,
            "valid_targets_per_optimizer_update": args.candidate * sequence_length * accumulation,
            "startup_loss": startup_loss,
            "measured_losses": losses,
            "startup_peak_allocated_bytes": startup_allocated,
            "startup_peak_reserved_bytes": startup_reserved,
            "steady_peak_allocated_bytes": steady_allocated,
            "steady_peak_reserved_bytes": steady_reserved,
            "fit_limit_reserved_bytes": limit,
            "total_device_memory_bytes": properties.total_memory,
            "gpu_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
        })
    except (torch.OutOfMemoryError, RuntimeError, FloatingPointError) as error:
        if isinstance(error, FloatingPointError):
            payload.update({"status": "numerical_failure", "error": repr(error)})
        elif isinstance(error, RuntimeError) and "out of memory" not in str(error).lower():
            payload.update({"status": "error", "error": repr(error)})
        else:
            payload.update({"status": "oom", "error": repr(error)})
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["torch_version"] = torch.__version__
    payload["cuda_runtime"] = torch.version.cuda
    write_json_atomic(args.result_path, payload)
    return 0


def run_candidate(args: argparse.Namespace, batch: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="batch-capacity-") as directory:
        result_path = Path(directory) / "result.json"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--config", str(args.config.resolve()),
            "--candidate", str(batch),
            "--repetitions", str(args.repetitions),
            "--headroom-fraction", str(args.headroom_fraction),
            "--result-path", str(result_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if not result_path.exists():
            return {
                "physical_batch_size": batch,
                "status": "process_error",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        result = json.loads(result_path.read_text())
        result["command"] = command
        return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.candidate is not None:
        if args.result_path is None:
            raise ValueError("candidate mode requires --result-path")
        raise SystemExit(candidate_run(args, config))
    if args.max_batch < 1 or args.repetitions < 1:
        raise ValueError("max batch and repetitions must be positive")

    attempts: list[dict] = []
    passed = 0
    failed = args.max_batch + 1
    batch = 1
    while batch <= args.max_batch:
        result = run_candidate(args, batch)
        attempts.append(result)
        print(json.dumps(result), flush=True)
        write_json_atomic(args.output.with_suffix(".attempts.json"), {"attempts": attempts})
        if result["status"] not in {"pass", "oom", "insufficient_headroom"}:
            raise RuntimeError(f"capacity probe failed for a non-memory reason: {result}")
        if result["status"] == "pass":
            passed = batch
            if batch == args.max_batch:
                break
            batch = min(args.max_batch, batch * 2)
            if batch == passed:
                break
        else:
            failed = batch
            break
    if passed == 0:
        raise RuntimeError("no physical batch size passed the capacity protocol")

    while failed - passed > 1:
        batch = (passed + failed) // 2
        result = run_candidate(args, batch)
        attempts.append(result)
        print(json.dumps(result), flush=True)
        write_json_atomic(args.output.with_suffix(".attempts.json"), {"attempts": attempts})
        if result["status"] not in {"pass", "oom", "insufficient_headroom"}:
            raise RuntimeError(f"capacity probe failed for a non-memory reason: {result}")
        if result["status"] == "pass":
            passed = batch
        else:
            failed = batch

    first = next(item for item in attempts if item["status"] == "pass")
    report = {
        "protocol": "fresh process per candidate; one optimizer-state initialization update; "
                    f"{args.repetitions} repeated complete updates; reserved-memory headroom gate",
        "selected_physical_batch_size": passed,
        "search_was_capped": passed == args.max_batch,
        "max_batch_tested": args.max_batch,
        "headroom_fraction": args.headroom_fraction,
        "repetitions": args.repetitions,
        "method": config.get("method", "baseline"),
        "sequence_length": config["sequence_length"],
        "precision": config["precision"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "model": config["model"],
        "parameter_count": BaselineLM(ModelConfig(**config["model"])).parameter_count(),
        "gpu_name": first.get("gpu_name"),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "attempts": attempts,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({"capacity_report": str(args.output), **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
