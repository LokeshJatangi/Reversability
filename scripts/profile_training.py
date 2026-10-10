#!/usr/bin/env python3
"""Separate timing, operator traces and allocator diagnostics; never resume training."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reversibility import build_model
import reversibility.reversible as reversible
from train_baseline import batch_at, file_sha256, runtime_metadata

FROZEN_MANIFEST = "71a12f5765cf37b7f5fb8753d51fa0828915a6db313ae20aef7fa9fb2e14df8a"


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Regions:
    """Nested peaks survive child resets; synchronized diagnostics are not benchmarks."""
    def __init__(self, device, mode):
        self.device, self.mode = device, mode
        self.rows, self.active = [], []
        self.peak_allocated = self.peak_reserved = 0

    def sample(self):
        if self.device.type != "cuda":
            return
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        for row in self.active:
            row["peak_allocated_bytes"] = max(row["peak_allocated_bytes"], allocated)
            row["peak_reserved_bytes"] = max(row["peak_reserved_bytes"], reserved)

    @contextlib.contextmanager
    def region(self, name):
        if self.mode == "timing":
            yield
            return
        if self.mode == "trace":
            with torch.profiler.record_function("profile/" + name):
                yield
            return
        sync(self.device)
        self.sample()
        row = {"region": name, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
        if self.device.type == "cuda":
            row["start_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
            row["start_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.active.append(row)
        try:
            yield
        finally:
            sync(self.device)
            self.sample()
            self.active.pop()
            if self.device.type == "cuda":
                row["end_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
                row["end_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            self.rows.append(row)


@contextlib.contextmanager
def annotations(model, regions):
    """Temporary wrappers only: preserve the original frozen model/backward code."""
    undo = []
    def wrap(owner, name, label):
        original = getattr(owner, name)
        own_attribute = name in owner.__dict__
        @functools.wraps(original)
        def annotated(*args, **kwargs):
            actual_label = label() if callable(label) else label
            with regions.region(actual_label):
                return original(*args, **kwargs)
        setattr(owner, name, annotated)
        def restore():
            if own_attribute:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        undo.append(restore)
    try:
        for block in model.blocks:
            wrap(block.attn, "forward", "component/attention_forward")
            wrap(block.mlp, "forward", "component/mlp_forward")
        wrap(model.token_embedding, "forward", "component/token_embedding")
        wrap(model.position_embedding, "forward", "component/position_embedding")
        wrap(model.final_norm, "forward", "component/final_norm")
        wrap(model.lm_head, "forward", "component/output_projection")
        if getattr(model, "backward_mode", None) == "reconstruct":
            wrap(reversible, "inverse_pair", "reversible/inverse_reconstruction")
            wrap(reversible, "forward_pair", lambda: "reversible/local_recompute"
                 if torch.is_grad_enabled() else "reversible/core_forward")
            wrap(torch.autograd, "grad", "reversible/local_vjp")
        yield
    finally:
        for restore in reversed(undo):
            restore()


def optimizer_update(model, optimizer, scaler, train, config, cursor, device, regions):
    batch = config["physical_batch_size"]
    length = config["sequence_length"]
    accumulation = config["gradient_accumulation_steps"]
    count = batch * length * accumulation
    amp = (lambda: torch.autocast("cuda", dtype={"fp16": torch.float16,
            "bf16": torch.bfloat16}[config["precision"]])) if device.type == "cuda" and config["precision"] != "fp32" else contextlib.nullcontext
    with regions.region("update/zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    processed = 0
    for _ in range(accumulation):
        with regions.region("update/data_and_transfer"):
            x, y, valid = batch_at(train, cursor + processed, batch, length, cursor + count)
            x, y = x.to(device), y.to(device)
        with amp():
            with regions.region("update/model_forward"):
                logits = model(x)
            with regions.region("update/cross_entropy"):
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                       ignore_index=-100, reduction="sum")
                normalized = loss / count
            with regions.region("update/backward"):
                if scaler is None:
                    normalized.backward()
                else:
                    scaler.scale(normalized).backward()
        # Match the trainer's scalar extraction/lifetime before the next microstep.
        with regions.region("update/loss_scalar"):
            loss_total += loss.detach().float().item()
        processed += valid
    with regions.region("update/unscale_and_clip"):
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    with regions.region("update/schedule_and_optimizer"):
        progress = (cursor + count) / config["max_train_targets"]
        warmup = min(1., (cursor + count) / config["warmup_targets"])
        cosine = config["min_learning_rate"] + .5 * (config["learning_rate"] - config["min_learning_rate"]) * (1 + math.cos(math.pi * progress))
        lr = min(config["learning_rate"] * warmup, cosine)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if scaler is None:
            optimizer.step()
        else:
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < before:
                raise FloatingPointError("AMP overflow: profiling update was skipped; no throughput claim")
    return {"loss": loss_total / count, "gradient_norm": norm.detach(), "targets": processed}


def check_update(result):
    if not math.isfinite(result["loss"]) or not bool(torch.isfinite(result["gradient_norm"])):
        raise FloatingPointError("Nonfinite loss/gradient in profiling probe")


def memory_stats(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def event_row(event):
    return {"name": event.key, "count": event.count,
            "cpu_total_us": event.cpu_time_total, "self_cpu_us": event.self_cpu_time_total,
            "device_total_us": getattr(event, "device_time_total", 0.),
            "self_device_us": getattr(event, "self_device_time_total", 0.),
            "self_device_memory_delta_bytes": getattr(event, "self_device_memory_usage", 0),
            "input_shapes": event.input_shapes}


def worker(args):
    config = json.loads(args.config.read_text())
    config.update(method=args.method, physical_batch_size=args.batch,
                  data_dir=str(args.data_dir), compile=False)
    if args.method == "midpoint":
        config.update(backward_mode="reconstruct", step_size=.5)
    if args.cpu_smoke:
        config["precision"] = "fp32"
    device = torch.device("cpu" if args.cpu_smoke else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. --cpu-smoke is plumbing validation only.")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "mode": args.mode, "command": shlex.join(sys.argv),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "config_file_sha256": file_sha256(args.config),
              "environment": runtime_metadata(device), "cpu_smoke_only": args.cpu_smoke}
    save(args.output / "result.json", result)
    try:
        if config["precision"] not in ("fp16", "bf16", "fp32"):
            raise ValueError("Unsupported precision")
        if config["sequence_length"] > config["model"]["max_sequence_length"]:
            raise ValueError("Context exceeds learned position embedding length")
        if not args.cpu_smoke and config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported on this GPU")
        manifest_path = args.data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        result["manifest_sha256"] = file_sha256(manifest_path)
        if not args.cpu_smoke and result["manifest_sha256"] != FROZEN_MANIFEST:
            raise ValueError("Not the frozen study dataset manifest")
        path = args.data_dir / "train.bin"
        if path.stat().st_size != (manifest["training_targets"] + 1) * 2 or file_sha256(path) != manifest["files"]["train"]["sha256"]:
            raise ValueError("Training binary size/hash mismatch")
        train = np.memmap(path, dtype=np.uint16, mode="r")
        steps = args.updates if args.mode == "timing" else args.trace_updates if args.mode == "trace" else 1
        targets_per_update = args.batch * config["sequence_length"] * config["gradient_accumulation_steps"]
        if targets_per_update * (args.warmup + steps) > min(manifest["training_targets"], config["max_train_targets"]):
            raise ValueError("Probe would exceed the available frozen training targets")
        random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        model = build_model(config).to(device).train()
        result["parameters"] = model.parameter_count()
        result["source_sha256"] = {str(p.relative_to(ROOT)): file_sha256(p) for p in
            [ROOT / "scripts/profile_training.py", ROOT / "scripts/train_baseline.py", *sorted((ROOT / "src/reversibility").glob("*.py"))]}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                  betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config["precision"] == "fp16" else None
        plain = Regions(device, "timing")
        cursor = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup):
            check_update(optimizer_update(model, optimizer, scaler, train, config, cursor, device, plain))
            cursor += targets_per_update
        sync(device)
        result["warmup"] = {"updates": args.warmup, "targets": cursor, **memory_stats(device)}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        results = []
        if args.mode == "timing":
            durations = []
            for _ in range(steps):
                sync(device); start = time.perf_counter()
                item = optimizer_update(model, optimizer, scaler, train, config, cursor, device, plain)
                sync(device); durations.append(time.perf_counter() - start)
                check_update(item); results.append(item["loss"])
                cursor += targets_per_update
            result["timing"] = {"update_seconds": durations, "total_seconds": sum(durations),
                "mean_seconds": statistics.mean(durations), "median_seconds": statistics.median(durations),
                "stdev_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.,
                "targets_per_second": steps * targets_per_update / sum(durations),
                "measured_valid_targets": steps * targets_per_update, **memory_stats(device)}
        elif args.mode == "trace":
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            regions = Regions(device, "trace")
            with annotations(model, regions), torch.profiler.profile(activities=activities,
                    record_shapes=True, profile_memory=True, with_stack=True) as profile:
                for _ in range(steps):
                    item = optimizer_update(model, optimizer, scaler, train, config, cursor, device, regions)
                    check_update(item); results.append(item["loss"])
                    cursor += targets_per_update
                sync(device)
            profile.export_chrome_trace(str(args.output / "trace.json"))
            events = [event_row(e) for e in profile.key_averages(group_by_input_shape=True)]
            save(args.output / "operators.json", events)
            result["regions"] = [e for e in events if e["name"].startswith("profile/")]
            sort_key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
            (args.output / "operators.txt").write_text(profile.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=50))
            result["trace_has_cuda_events"] = any(e.device_type == torch.autograd.DeviceType.CUDA for e in profile.events())
            if device.type == "cuda" and not result["trace_has_cuda_events"]:
                raise RuntimeError("CUDA kernels were not captured; profiler/CUPTI is unavailable")
            result["instrumented_memory"] = memory_stats(device)
        else:
            regions = Regions(device, "memory")
            recording = False
            try:
                if device.type == "cuda":
                    torch.cuda.memory._record_memory_history(max_entries=100000)
                    recording = True
                with annotations(model, regions):
                    item = optimizer_update(model, optimizer, scaler, train, config, cursor, device, regions)
                    check_update(item); results.append(item["loss"])
                sync(device); regions.sample()
                if recording:
                    torch.cuda.memory._dump_snapshot(str(args.output / "allocator_snapshot.pickle"))
            finally:
                if recording:
                    torch.cuda.memory._record_memory_history(enabled=None)
            result["memory_regions"] = regions.rows
            result["memory_diagnostic_peak"] = {"peak_allocated_bytes": regions.peak_allocated if device.type == "cuda" else None,
                                                "peak_reserved_bytes": regions.peak_reserved if device.type == "cuda" else None}
        result.update(status="complete", losses=results, measured_updates=steps,
            valid_targets_per_update=targets_per_update, effective_batch_sequences=args.batch * config["gradient_accumulation_steps"],
            note="Fresh probe, no evaluation/checkpoint I/O; targets are separate from completed full runs. Trace rows are inclusive and overlap. Memory region peaks are total live allocation, not component-owned activation bytes.")
        result["logits_tensor"] = {"shape": [args.batch, config["sequence_length"], config["model"]["vocab_size"]],
            "fp16_bytes": args.batch * config["sequence_length"] * config["model"]["vocab_size"] * 2,
            "same_shape_fp32_bytes": args.batch * config["sequence_length"] * config["model"]["vocab_size"] * 4}
    except Exception as error:
        result.update(status="failed", error=repr(error))
        save(args.output / "result.json", result)
        raise
    save(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output), "mode": args.mode}), flush=True)


def suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    scenarios = [("baseline_29", "baseline", 29), ("midpoint_29", "midpoint", 29), ("midpoint_33", "midpoint", 33)]
    if args.cpu_smoke:
        scenarios = [("baseline_2", "baseline", 2), ("midpoint_2", "midpoint", 2), ("midpoint_3", "midpoint", 3)]
    records = []
    for name, method, batch in scenarios:
        for mode in ("timing", "trace", "memory"):
            destination = args.output / name / mode
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(args.config),
                "--data-dir", str(args.data_dir), "--output", str(destination), "--method", method,
                "--batch", str(batch), "--mode", mode, "--warmup", str(args.warmup),
                "--updates", str(args.updates), "--trace-updates", str(args.trace_updates)]
            if args.cpu_smoke:
                command.append("--cpu-smoke")
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Profiling {name}: {mode} (fresh process)", flush=True)
            with (destination.parent / f"{mode}.log").open("w") as log:
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            record_path = destination / "result.json"
            record = json.loads(record_path.read_text()) if record_path.exists() else {"status": "failed", "error": "No worker result; inspect log"}
            records.append({"scenario": name, "mode": mode, "returncode": code, "result": record})
            save(args.output / "suite.json", {"status": "running", "cpu_smoke_only": args.cpu_smoke, "records": records})
            print(f"  {record['status']}: {destination}", flush=True)
    complete = all(r["returncode"] == 0 and r["result"]["status"] == "complete" for r in records)
    save(args.output / "suite.json", {"status": "complete" if complete else "partial_failure", "cpu_smoke_only": args.cpu_smoke, "records": records})
    lines = ["# Training-path profiling", "", "CPU smoke only; no CUDA conclusions." if args.cpu_smoke else "CUDA probes; training-only timings exclude evaluation and checkpoint I/O.", "", "| Scenario | Valid targets/s | Median update seconds | Steady allocated GiB | Steady reserved GiB |", "|---|---:|---:|---:|---:|"]
    for r in records:
        if r["mode"] != "timing" or r["result"]["status"] != "complete":
            continue
        t = r["result"]["timing"]
        gib = lambda x: f"{x / 2**30:.3f}" if x is not None else "unavailable"
        lines.append(f"| {r['scenario']} | {t['targets_per_second']:.1f} | {t['median_seconds']:.4f} | {gib(t['peak_allocated_bytes'])} | {gib(t['peak_reserved_bytes'])} |")
    lines += ["", "Matched batch-29 probes isolate the method comparison. Batch 33 uses effective batch 66 rather than 58 and is a separate throughput configuration.", "", "## Trace and memory interpretation", "", "Operator self-device time ranks kernels without adding nested inclusive regions. component/attention_forward and component/mlp_forward include forward, inverse and recomputation calls. Backward kernels remain visible in the operator trace; forward ranges do not represent all attention/MLP gradient time.", "", "reversible/inverse_reconstruction, local_recompute and local_vjp are separately marked within backward. Inclusive times overlap with component ranges and must not be summed. Region memory peaks include all tensors live at that time; do not sum them or call them activation-only savings.", "", "Use uninstrumented timing peaks as the reference. Stack/shape profiling retains extra tensor references; instrumented trace peaks can be inflated or fail at a previously fitting batch. Allocator snapshots and synchronized phase-memory probes are diagnostic, not timing benchmarks.", "", "Inspect output_projection, cross_entropy, logits-shaped casts/log-softmax buffers, backward GEMMs and reconstruction before choosing an optimization. A large calculated logits tensor alone does not prove it dominates the measured peak.", "", f"Suite status: {'complete' if complete else 'partial_failure; inspect failed mode logs, preserve successful outputs'}. No full training run or resumed checkpoint was modified."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if not complete:
        raise RuntimeError("Some profiling passes failed; preserved results and logs in output directory")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", action="store_true")
    p.add_argument("--method", choices=["baseline", "midpoint"], default="baseline")
    p.add_argument("--batch", type=int, default=29)
    p.add_argument("--mode", choices=["timing", "trace", "memory"], default="timing")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--trace-updates", type=int, default=1)
    p.add_argument("--cpu-smoke", action="store_true", help="Tiny CPU plumbing validation; never GPU evidence")
    a = p.parse_args()
    a.config = a.config.resolve()
    a.data_dir = a.data_dir.resolve()
    a.output = a.output.resolve()
    if min(a.warmup, a.updates, a.trace_updates, a.batch) < 1:
        p.error("Warmup, update counts and batch must be positive")
    config = json.loads(a.config.read_text())
    if min(config["sequence_length"], config["gradient_accumulation_steps"]) < 1:
        p.error("Context and accumulation must be positive")
    if config.get("compile"):
        p.error("Compiled profiling is not validated")
    if a.cpu_smoke and (config["model"]["d_model"] > 64 or config["sequence_length"] > 64):
        p.error("CPU smoke requires a tiny config, not the full CUDA model")
    return a


if __name__ == "__main__":
    args = parse_args()
    suite(args) if args.suite else worker(args)
