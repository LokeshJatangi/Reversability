#!/usr/bin/env python3
"""Correctness-gated short optimization experiments; not a full-training run."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reversibility import build_model
from reversibility.optimized import amp_context, chunked_backward, cpu_snapshot
from profile_training import FROZEN_MANIFEST, save, sync, memory_stats, Regions, optimizer_update
from train_baseline import batch_at, file_sha256, runtime_metadata, atomic_save
from validate_reversible import OPTIMIZER_TOLERANCES, UPDATE_RELATIVE_L2

HEADROOM = .10  # Fixed reserved-memory safety margin, never tuned to make a batch pass.
HISTORICAL_TPS = 56197.91609955422


def seed(config, device):
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])


def optimizer_for(model, config, fused=False):
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]), weight_decay=config["weight_decay"], fused=fused)


def microstep_batches(config):
    physical = config["physical_batch_size"]
    if physical < 1 or config.get("gradient_accumulation_steps", 1) != 1 or config.get("effective_batch_size", physical) != physical:
        raise ValueError("Capacity experiment requires one real physical batch, accumulation1")
    return [physical]


def targets_per_update(config):
    return sum(microstep_batches(config)) * config["sequence_length"]


def optimized_update(model, optimizer, scaler, train, config, cursor, device, chunk, reuse, loss_backend="torch"):
    optimizer.zero_grad(set_to_none=True)
    count = targets_per_update(config)
    total = torch.zeros((), device=device)
    processed = 0
    for micro_batch in microstep_batches(config):
        x, y, valid = batch_at(train, cursor + processed, micro_batch, config["sequence_length"], cursor + count)
        x, y = x.to(device), y.to(device)
        total += chunked_backward(model, x, y, count, chunk, scaler, config["precision"], reuse, loss_backend)
        processed += valid
    if processed != count:
        raise ValueError("Probe must contain the declared number of valid targets")
    if scaler is not None:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    progress = (cursor + count) / config["max_train_targets"]
    warmup = min(1., (cursor + count) / config["warmup_targets"])
    cosine = config["min_learning_rate"] + .5 * (config["learning_rate"] - config["min_learning_rate"]) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = min(config["learning_rate"] * warmup, cosine)
    if scaler is None:
        optimizer.step()
    else:
        scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < scale:
            raise FloatingPointError("AMP overflow; no speed claim")
    # Single loss extraction per complete update; no chunk-level device sync.
    return {"loss": total.item() / count, "gradient_norm": norm.detach(), "targets": processed}


def correctness_gate(config, device, fused=False, loss_backend="torch", swiglu=False, reuse=False):
    """Two single-microstep updates against frozen forward/CE, including tied weights/masks.

    Same full vocabulary/depth; CUDA B8/context512 gates all candidate chunks.
    CPU uses B2 and bounded context64 for local execution.
    Fixed FP16 gradient L2 tolerance 2%, loss tolerance .002; optimizer
    per-step coordinate tolerance .0012 and whole-update relative L2 .05
    inherit frozen v2. Cumulative parameter differences are reported separately.
    FP32 limits: gradient L2 .00005, loss .000002; update coordinate
    policy and whole-update relative L2 are inherited from frozen policy v2.
    """
    config = copy.deepcopy(config)
    precision = config["precision"]
    amp = precision != "fp32" and device.type == "cuda"
    limits = {"gradient_relative_l2": .02 if amp else 5e-5,
              "loss_relative": .002 if amp else 2e-6,
              "optimizer_step_max_absolute": OPTIMIZER_TOLERANCES["fp16" if amp else "fp32"]["atol"],
              "optimizer_relative_update_l2": UPDATE_RELATIVE_L2["fp16" if amp else "fp32"]}
    records = []
    for method in ("baseline", "midpoint"):
        for gate_chunk in ((37,) if device.type == "cpu" else (1024, 2048, 4096)):
            config.update(method=method, backward_mode="reconstruct", step_size=.5, compile=False)
            seed(config, device)
            original = build_model(config).to(device)
            candidate = copy.deepcopy(original)
            if swiglu:
                from reversibility.kernels import enable_swiglu
                enable_swiglu(candidate)
            left = optimizer_for(original, config)
            right = optimizer_for(candidate, config, fused)
            # Fixed scale validates scaling without requiring two independently adapting scalers.
            scale = 128. if amp else 1.
            class FixedScale:
                def scale(self, loss): return loss * scale
            for update in range(2):
                left.zero_grad(set_to_none=True); right.zero_grad(set_to_none=True)
                reference_loss = torch.zeros((), device=device)
                candidate_loss = torch.zeros((), device=device)
                for micro in range(1):
                    gate_batch = 2 if device.type == "cpu" else 8
                    gate_length = min(64, config["sequence_length"]) if device.type == "cpu" else config["sequence_length"]
                    ids = torch.randint(config["model"]["vocab_size"], (gate_batch, gate_length), device=device)
                    labels = ids.roll(-1, 1).clone()
                    labels[:, -3:] = -100
                    denominator = int((labels != -100).sum())
                    with amp_context(device, precision):
                        loss = F.cross_entropy(original(ids).flatten(0,1), labels.flatten(), ignore_index=-100, reduction="sum")
                        (loss * scale / denominator).backward()
                    reference_loss += loss.detach().float()
                    candidate_loss += chunked_backward(candidate, ids, labels, denominator, gate_chunk,
                        FixedScale(), precision, reuse=reuse and method == "midpoint", loss_backend=loss_backend)
                grad_errors = []
                for a, b in zip(original.parameters(), candidate.parameters()):
                    a.grad.div_(scale); b.grad.div_(scale)
                    if not torch.isfinite(a.grad).all() or not torch.isfinite(b.grad).all():
                        raise FloatingPointError("Nonfinite correctness-gate gradient")
                    grad_errors.append(float((a.grad - b.grad).norm() / a.grad.norm().clamp_min(1e-12)))
                torch.nn.utils.clip_grad_norm_(original.parameters(), config["grad_clip"])
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), config["grad_clip"])
                before = [p.detach().clone() for p in original.parameters()]
                candidate_before = [p.detach().clone() for p in candidate.parameters()]
                left.step(); right.step()
                difference_sq = sum((a.detach()-b.detach()).double().square().sum().item() for a,b in zip(original.parameters(),candidate.parameters()))
                update_sq = sum((a.detach()-old).double().square().sum().item() for a,old in zip(original.parameters(),before))
                row = {"method": method, "update": update, "chunk_tokens": gate_chunk,
                    "gate_batch": gate_batch, "gate_context": gate_length,
                    "gradient_relative_l2": max(grad_errors),
                    "loss_relative": float((reference_loss-candidate_loss).abs()/reference_loss.abs().clamp_min(1e-12)),
                    "parameter_max_absolute": max(float((a-b).detach().abs().max()) for a,b in zip(original.parameters(),candidate.parameters())),
                    "optimizer_step_max_absolute": max(float(((a.detach()-old_a)-(b.detach()-old_b)).abs().max()) for a,b,old_a,old_b in zip(original.parameters(),candidate.parameters(),before,candidate_before)),
                    "optimizer_relative_update_l2": math.sqrt(difference_sq/max(update_sq,1e-30))}
                row["passed"] = all(row[k] <= v for k,v in limits.items())
                records.append(row)
            del original, candidate, left, right, before, candidate_before
    passing_chunks = [chunk for chunk in sorted({r["chunk_tokens"] for r in records}) if all(r["passed"] for r in records if r["chunk_tokens"]==chunk)]
    return {"limits": limits, "records": records, "passing_chunks":passing_chunks, "force_reuse":reuse, "passed": all(r["passed"] for r in records),
            "fused_optimizer": fused, "loss_backend":loss_backend, "fused_swiglu":swiglu, "probe_grad_scale": scale, "optimizer_policy": "frozen-v2-per-step-coordinate-and-whole-update-L2"}


def checkpoint_cpu(model, optimizer, scaler, config, cursor, manifest_hash, directory, device):
    sync(device)
    start = time.perf_counter()
    payload = cpu_snapshot({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": config, "config_sha256": __import__("hashlib").sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        "manifest_sha256": manifest_hash, "committed_targets": cursor,
        "target_budget": config["max_train_targets"], "probe_only": True,
        "step": cursor // targets_per_update(config),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "grad_scaler": None if scaler is None else scaler.state_dict()})
    sync(device)
    copy_seconds = time.perf_counter() - start
    start = time.perf_counter()
    path = directory / "probe-checkpoint-cpu.pt"
    atomic_save(payload, path)
    disk_seconds = time.perf_counter() - start
    restored = torch.load(path, map_location="cpu", weights_only=False)
    def tensors(value):
        if isinstance(value, torch.Tensor): yield value
        elif isinstance(value, dict):
            for item in value.values(): yield from tensors(item)
        elif isinstance(value, (tuple,list)):
            for item in value: yield from tensors(item)
    aa, bb = list(tensors(payload)), list(tensors(restored))
    verified = len(aa)==len(bb) and all(a.device.type==b.device.type=="cpu" and torch.equal(a,b) for a,b in zip(aa,bb))
    if not verified: raise RuntimeError("CPU checkpoint roundtrip failed")
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        "copy_to_cpu_seconds": copy_seconds, "disk_write_seconds": disk_seconds,
        "roundtrip_verified": verified, "included_in_training_timing": False, **memory_stats(device)}


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cpu" if args.cpu_smoke else "cuda")
    gpu_monitor = None
    monitor_file = None
    result = {"status":"running", "cpu_smoke_only":args.cpu_smoke, "command":sys.argv,
              "timestamp_utc":datetime.now(timezone.utc).isoformat(), "headroom":HEADROOM}
    save(args.output / "result.json", result)
    try:
        if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA required")
        config=json.loads(args.config.read_text())
        # Physical-capacity experiment: exactly one microstep per optimizer update.
        reference_batch = config["physical_batch_size"]
        config.update(method=args.method,physical_batch_size=args.batch,compile=False,backward_mode="reconstruct",step_size=.5,
                      gradient_accumulation_steps=1, effective_batch_size=args.batch)
        config["microstep_batch_sizes"] = microstep_batches(config)
        if config["microstep_batch_sizes"] != [args.batch]:
            raise ValueError("This experiment requires one physical batch, no accumulation")
        if not args.chunk and not args.gate and args.batch != reference_batch:
            raise ValueError("Unoptimized reference must use the original physical batch29 (tiny CPU reference2)")
        if args.cpu_smoke: config["precision"]="fp32"
        result.update(config=config, chunk_tokens=args.chunk, force_reuse=args.reuse, fused_optimizer=args.fused,
                      loss_backend=args.loss_backend, fused_swiglu=args.swiglu, allocator=args.allocator, allocator_environment=os.environ.get("PYTORCH_ALLOC_CONF",os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
                      environment=runtime_metadata(device), config_file_sha256=file_sha256(args.config))
        result["source_sha256"] = {str(p.relative_to(ROOT)):file_sha256(p) for p in
            [Path(__file__), ROOT/"scripts/profile_training.py",ROOT/"scripts/train_baseline.py",*sorted((ROOT/"src/reversibility").glob("*.py"))]}
        if args.gate:
            result["gate"]=correctness_gate(config,device,args.fused,args.loss_backend,args.swiglu,args.reuse)
            if not result["gate"]["passed"]: raise RuntimeError("Numerical gate failed; sweep blocked")
        else:
            manifest_path=args.data_dir/"manifest.json"
            manifest=json.loads(manifest_path.read_text()); manifest_hash=file_sha256(manifest_path)
            if not args.cpu_smoke and manifest_hash!=FROZEN_MANIFEST: raise ValueError("Frozen manifest mismatch")
            path=args.data_dir/"train.bin"
            if path.stat().st_size!=(manifest["training_targets"]+1)*2 or file_sha256(path)!=manifest["files"]["train"]["sha256"]: raise ValueError("Training binary mismatch")
            result["manifest_sha256"]=manifest_hash
            train=np.memmap(path,dtype=np.uint16,mode="r")
            count=targets_per_update(config)
            if count*(args.warmup+args.updates)>min(manifest["training_targets"],config["max_train_targets"]): raise ValueError("Insufficient frozen targets")
            seed(config,device)
            model=build_model(config).to(device).train()
            result["parameters"]=model.parameter_count()
            if args.swiglu:
                from reversibility.kernels import enable_swiglu
                enable_swiglu(model)
            optimizer=optimizer_for(model,config,args.fused)
            scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" and config["precision"]=="fp16" else None
            regions=Regions(device,"timing")
            def run(cursor):
                if args.chunk:
                    item=optimized_update(model,optimizer,scaler,train,config,cursor,device,args.chunk,args.reuse,args.loss_backend)
                else:
                    item=optimizer_update(model,optimizer,scaler,train,config,cursor,device,regions)
                if not math.isfinite(item["loss"]) or not torch.isfinite(item["gradient_norm"]): raise FloatingPointError("Nonfinite loss/gradient")
                return item
            if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
            for update in range(args.warmup): run(update*count)
            sync(device)
            result["startup_memory"]=memory_stats(device)
            if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
            if device.type=="cuda" and shutil.which("nvidia-smi"):
                monitor_file=(args.output/"gpu-utilization.csv").open("w")
                gpu_monitor=subprocess.Popen(["nvidia-smi","--query-gpu=utilization.gpu,utilization.memory,memory.used","--format=csv,noheader,nounits","--loop-ms=500"],stdout=monitor_file,stderr=subprocess.DEVNULL)
            durations=[]; losses=[]
            for update in range(args.updates):
                sync(device); start=time.perf_counter()
                item=run((args.warmup+update)*count)
                sync(device); durations.append(time.perf_counter()-start); losses.append(item["loss"])
            result["timing"]={"update_seconds":durations,"losses":losses,"targets_per_second":args.updates*count/sum(durations),
                "mean_update_seconds":statistics.mean(durations),"measured_valid_targets":args.updates*count,**memory_stats(device)}
            if gpu_monitor is not None:
                gpu_monitor.terminate(); gpu_monitor.wait(timeout=10); monitor_file.close()
                gpu_monitor=None; monitor_file=None
                samples=[]
                for line in (args.output/"gpu-utilization.csv").read_text().splitlines():
                    try: samples.append([float(x.strip()) for x in line.split(",")])
                    except ValueError: pass
                result["gpu_utilization"]={"sampling_ms":500,"samples":samples,"mean_gpu_busy_percent":statistics.mean(x[0] for x in samples) if samples else None,
                    "note":"Coarse device-busy samples during timed training; memory utilization is controller activity, not allocated capacity or SM occupancy"}
            result["effective_batch_size"]=sum(microstep_batches(config))
            result["microstep_batch_sizes"]=microstep_batches(config)
            result["valid_targets_per_update"]=count
            result["ordered_target_range"]=[0,count*(args.warmup+args.updates)]
            if device.type=="cuda":
                total=torch.cuda.get_device_properties(device).total_memory
                reserve=max(result["startup_memory"]["peak_reserved_bytes"],result["timing"]["peak_reserved_bytes"])
                result["capacity"]={"total_device_bytes":total,"peak_reserved_bytes":reserve,
                    "reserved_fraction":reserve/total,
                    "peak_allocated_bytes":max(result["startup_memory"]["peak_allocated_bytes"],result["timing"]["peak_allocated_bytes"]),
                    "allocated_fraction":max(result["startup_memory"]["peak_allocated_bytes"],result["timing"]["peak_allocated_bytes"])/total,
                    "unused_reserved_budget_bytes":int((1-HEADROOM)*total)-reserve,
                    "passed":reserve <= (1-HEADROOM)*total}
            else: result["capacity"]={"passed":None,"reason":"CPU smoke cannot establish CUDA capacity"}
            if args.checkpoint:
                result["checkpoint"]=checkpoint_cpu(model,optimizer,scaler,config,count*(args.warmup+args.updates),manifest_hash,args.output,device)
                save(args.output/"checkpoint-overhead.json",result["checkpoint"])
                if device.type=="cuda":
                    peak=result["checkpoint"]["peak_reserved_bytes"]
                    result["capacity"]["passed"] &= peak <= (1-HEADROOM)*total
                    result["capacity"]["peak_reserved_bytes"]=max(reserve,peak)
                    result["capacity"]["reserved_fraction"]=max(reserve,peak)/total
                    result["capacity"]["unused_reserved_budget_bytes"]=int((1-HEADROOM)*total)-max(reserve,peak)
                    result["capacity"]["peak_allocated_bytes"]=max(result["capacity"]["peak_allocated_bytes"],result["checkpoint"]["peak_allocated_bytes"])
                    result["capacity"]["allocated_fraction"]=result["capacity"]["peak_allocated_bytes"]/total
        result["status"]="complete"
    except Exception as error:
        result.update(status="failed",error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        if device.type=="cuda" and torch.cuda.is_available():
            result["failure_memory"]=memory_stats(device)
        raise
    finally:
        if gpu_monitor is not None:
            gpu_monitor.terminate(); gpu_monitor.wait(timeout=10)
        if monitor_file is not None: monitor_file.close()
        save(args.output/"result.json",result)


def suite(args):
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    def invoke(name,method="baseline",batch=29,chunk=0,reuse=False,fused=False,gate=False,checkpoint=False,loss_backend="torch",swiglu=False,allocator="default",short=False):
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--config",str(args.config),"--data-dir",str(args.data_dir),
            "--output",str(args.output/name),"--method",method,"--batch",str(batch),"--chunk",str(chunk),
            "--warmup",str(3 if short else args.warmup),"--updates",str(3 if short else args.updates),"--loss-backend",loss_backend,"--allocator",allocator]
        for flag, enabled in [("--reuse",reuse),("--fused",fused),("--gate",gate),("--checkpoint",checkpoint),("--swiglu",swiglu),("--cpu-smoke",args.cpu_smoke)]:
            if enabled: command.append(flag)
        print("RUN",name,flush=True)
        with (args.output/(name+".log")).open("w") as log:
            task_env=os.environ.copy()
            task_env.pop("PYTORCH_ALLOC_CONF",None); task_env.pop("PYTORCH_CUDA_ALLOC_CONF",None)
            if allocator=="expandable": task_env["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=task_env)
        result_path=args.output/name/"result.json"
        row={"name":name,"returncode":process.returncode,"result":json.loads(result_path.read_text()) if result_path.exists() else {"status":"failed"}}
        records.append(row); save(args.output/"suite.json",{"records":records,"headroom":HEADROOM})
        print("RESULT",name,row["result"]["status"],flush=True)
        return row
    gate_row=invoke("correctness-gate",gate=True)
    allowed_chunks=gate_row["result"].get("gate",{}).get("passing_chunks",[])
    if not allowed_chunks: raise RuntimeError("No shared numerical gate passed; no optimized benchmarks launched")
    reuse_row=invoke("force-reuse-gate",method="midpoint",gate=True,reuse=True)
    reuse_chunks=set(reuse_row["result"].get("gate",{}).get("passing_chunks",[]))
    fused_chunks=set()
    if not args.cpu_smoke:
        row=invoke("fused-correctness-gate",gate=True,fused=True)
        fused_chunks=set(row["result"].get("gate",{}).get("passing_chunks",[]))
    kernel_options=[]
    if not args.cpu_smoke:
        for loss_backend,swiglu,reuse,label in [("triton",False,False,"ce"),("torch",True,False,"swiglu"),("triton",True,False,"both"),("triton",True,True,"both-reuse")]:
            row=invoke("kernel-gate-"+label,method="midpoint" if reuse else "baseline",gate=True,loss_backend=loss_backend,swiglu=swiglu,reuse=reuse)
            passing=set(row["result"].get("gate",{}).get("passing_chunks",[])) & set(allowed_chunks)
            if passing: kernel_options.append((loss_backend,swiglu,reuse,label,passing))
    # References bracket candidates to expose runtime drift.
    for rep in range(3): invoke("reference-start-"+str(rep),batch=2 if args.cpu_smoke else 29)
    candidates=[]
    batches=(2,3) if args.cpu_smoke else (29,40,50)
    chunks=(3,) if args.cpu_smoke else tuple(allowed_chunks)
    for method in ("baseline","midpoint"):
        # Matched ablation: shared chunking before midpoint graph reuse.
        if method=="midpoint": invoke("midpoint-chunk-only",method,batches[0],chunks[0])
        for batch in batches:
            for chunk in chunks:
                for fused in ([False,True] if chunk in fused_chunks and batch==batches[-1] else [False]):
                    name=f"{method}-b{batch}-c{chunk}-f{int(fused)}"
                    row=invoke(name,method,batch,chunk,method=="midpoint" and not fused and ((37 in reuse_chunks) if args.cpu_smoke else chunk in reuse_chunks),fused)
                    if row["result"].get("capacity",{}).get("passed") or (args.cpu_smoke and row["result"]["status"]=="complete"):
                        candidates.append(row)
    # Kernel ablations apply equally to baseline and midpoint, at matched29 and real50.
    if kernel_options:
        for method in ("baseline","midpoint"):
            for batch in (batches[0],batches[-1]):
                fitting=[r for r in candidates if r["result"]["config"]["method"]==method and r["result"]["config"]["physical_batch_size"]==batch]
                # Even if Torch50 OOMs, try the bounded kernel path at1024.
                chosen=min(fitting,key=lambda r:r["result"].get("capacity",{}).get("peak_reserved_bytes") or float("inf"))["result"] if fitting else {"chunk_tokens":1024,"fused_optimizer":False}
                for loss_backend,swiglu,reuse,label,passing in kernel_options:
                    if reuse and method!="midpoint": continue
                    kernel_chunk=chosen["chunk_tokens"] if chosen["chunk_tokens"] in passing else min(passing)
                    row=invoke(f"kernel-{method}-b{batch}-{label}",method,batch,kernel_chunk,reuse,False,loss_backend=loss_backend,swiglu=swiglu)
                    if row["result"].get("capacity",{}).get("passed"): candidates.append(row)
    # Allocator experiment runs in fresh processes, with its environment set before Torch imports.
    if not args.cpu_smoke:
        for method in ("baseline","midpoint"):
            fitting=[r for r in candidates if r["result"]["config"]["method"]==method and r["result"]["config"]["physical_batch_size"]==50]
            if fitting:
                chosen=min(fitting,key=lambda r:r["result"].get("capacity",{}).get("peak_reserved_bytes") or float("inf"))["result"]
                row=invoke("allocator-"+method,method,50,chosen["chunk_tokens"],chosen["force_reuse"],chosen["fused_optimizer"],loss_backend=chosen["loss_backend"],swiglu=chosen["fused_swiglu"],allocator="expandable")
                if row["result"].get("capacity",{}).get("passed"): candidates.append(row)
    for rep in range(3): invoke("reference-end-"+str(rep),batch=batches[0])
    # Re-test lowest-reserved-memory feasible batch-50 choice for each method, including CPU checkpoint.
    for method in ("baseline","midpoint"):
        eligible=[r for r in candidates if r["result"]["config"]["method"]==method and r["result"]["config"]["physical_batch_size"]==batches[-1]]
        if not eligible: continue
        chosen=min(eligible,key=lambda r:r["result"].get("capacity",{}).get("peak_reserved_bytes") or float("inf"))["result"]
        for rep in range(3):
            invoke(f"final-{method}-{rep}",method,batches[-1],chosen["chunk_tokens"],chosen["force_reuse"],chosen["fused_optimizer"],checkpoint=rep==2,loss_backend=chosen["loss_backend"],swiglu=chosen["fused_swiglu"],allocator=chosen["allocator"])
    # Capacity grows progressively; no accumulation/padding is used to imitate a batch.
    capacity_search={}
    if not args.cpu_smoke:
        for method in ("baseline","midpoint"):
            eligible=[r for r in candidates if r["result"]["config"]["method"]==method]
            if not eligible: continue
            largest_tested=max(r["result"]["config"]["physical_batch_size"] for r in eligible)
            eligible=[r for r in eligible if r["result"]["config"]["physical_batch_size"]==largest_tested]
            chosen=min(eligible,key=lambda r:r["result"].get("capacity",{}).get("peak_reserved_bytes") or float("inf"))["result"]
            lower=0; upper=None; trials=[]; search_error=None
            def capacity_trial(batch, suffix="", confirmation=False, checkpoint=False):
                nonlocal search_error
                row=invoke(f"capacity-{method}-b{batch}{suffix}",method,batch,chosen["chunk_tokens"],chosen["force_reuse"],chosen["fused_optimizer"],
                    loss_backend=chosen["loss_backend"],swiglu=chosen["fused_swiglu"],allocator=chosen["allocator"],short=not confirmation,checkpoint=checkpoint)
                trials.append(row["name"])
                if row["result"]["status"]!="complete" and "out of memory" not in row["result"].get("error", "").lower():
                    search_error={"trial":row["name"],"error":row["result"].get("error","Missing result")}
                return row["result"]["status"]=="complete" and row["result"].get("capacity",{}).get("passed",False)
            for batch in (29,33,36,40,44,48,50,64,96,128,192,256,384,512,768,1024):
                if capacity_trial(batch): lower=batch
                else:
                    if search_error is None: upper=batch
                    break
            if upper is not None and search_error is None:
                while upper-lower>1:
                    batch=(lower+upper)//2
                    if capacity_trial(batch,"-refine"): lower=batch
                    else:
                        if search_error is not None: break
                        upper=batch
            verified=None
            if lower and search_error is None:
                confirmations=[capacity_trial(lower,f"-confirm-{repeat}",confirmation=True,checkpoint=repeat==2) for repeat in range(3)]
                if all(confirmations): verified=lower
            capacity_search[method]={"largest_verified_physical_batch":verified,"tested_lower_bound":lower,
                "first_failing_physical_bound":upper,"search_ceiling":1024,"ceiling_reached":lower==1024 and upper is None and search_error is None, "search_error":search_error,
                "gradient_accumulation_steps":1,"trials":trials,
                "selected_optimization":{key:chosen[key] for key in ("chunk_tokens","force_reuse","fused_optimizer","loss_backend","fused_swiglu","allocator")},
                "validation_loss_ceiling":5.56229485, "full_training_targets":50_000_000,
                "note":"Memory-first configuration selection; three fresh full-duration confirmations including CPU checkpoint required; if1024 fits, this is a verified lower bound, not an absolute maximum. OOM/headroom bracket is refined assuming monotonic capacity for this fixed path; unexpected runtime failures are not memory boundaries. Capacity search is separate from batch50 speed acceptance."}
    refs=[r["result"]["timing"]["targets_per_second"] for r in records if r["name"].startswith("reference-") and r["result"]["status"]=="complete"]
    decisions={}
    for method in ("baseline","midpoint"):
        final=[r["result"] for r in records if r["name"].startswith("final-"+method+"-")]
        valid=len(final)==3 and all(r["status"]=="complete" and r.get("capacity",{}).get("passed") and r["config"]["gradient_accumulation_steps"]==1 and r["microstep_batch_sizes"]==[50] for r in final)
        rates=[r["timing"]["targets_per_second"] for r in final if r["status"]=="complete"]
        decisions[method]={"batch_50_verified":valid and not args.cpu_smoke,
            "median_targets_per_second":statistics.median(rates) if rates else None,
            "no_speed_loss_confirmed":valid and len(refs)==6 and min(rates)>=max(refs) and not args.cpu_smoke,
            "meets_historical_baseline":valid and min(rates)>=HISTORICAL_TPS and not args.cpu_smoke,
            "strict_rule":"Required: three batch50 repeats, accumulation1, reserved <=90%, numerical gates. Speed is optional; slowest finalist >= fastest baseline29 reference is recorded separately."}
    baseline_median=decisions["baseline"]["median_targets_per_second"]
    midpoint_median=decisions["midpoint"]["median_targets_per_second"]
    decisions["midpoint"]["throughput_ratio_vs_optimized_baseline_50"]=(midpoint_median/baseline_median if baseline_median and midpoint_median else None)
    report={"status":"cpu_smoke_complete" if args.cpu_smoke else "experiments_complete", "records":records,"decisions":decisions,
            "headroom":HEADROOM,"capacity_search":capacity_search,"reference_targets_per_second":refs,"historical_baseline_targets_per_second":HISTORICAL_TPS,
            "note":"Physical-capacity experiment: accumulation1 for every reference/candidate; baseline29 effective29, target50 effective50. Compare valid targets/s and memory, not raw update latency. Same frozen stream/target-indexed schedule, different targets per update and update counts; full-run quality/cost remains unmeasured. Historical56197.916 targets/s came from accumulation2 and is contextual only."}
    save(args.output/"suite.json",report)
    lines=["# Optimization experiments", "",report["note"],"", "Fixed reserved-memory headroom: 10%. CPU checkpoint copy/write is outside training timing.","",
           "| Experiment | Status | Targets/s | Allocated GiB | Reserved GiB | Capacity gate |", "|---|---|---:|---:|---:|---|"]
    for r in records:
        v=r["result"]; rate=v.get("timing",{}).get("targets_per_second")
        mem=v.get("timing",{})
        allocated=mem.get("peak_allocated_bytes"); reserved=mem.get("peak_reserved_bytes")
        lines.append(f"| {r['name']} | {v['status']} | {round(rate,2) if rate else '-'} | {round(allocated/2**30,3) if allocated else '-'} | {round(reserved/2**30,3) if reserved else '-'} | {v.get('capacity',{}).get('passed','-')} |")
    lines += ["", "Decisions (CPU values never establish GPU acceptance):", "", "```json",json.dumps({"batch50_decisions":decisions,"capacity_search":capacity_search},indent=2),"```"]
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n")


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--suite",action="store_true"); p.add_argument("--gate",action="store_true"); p.add_argument("--cpu-smoke",action="store_true")
    p.add_argument("--method",choices=["baseline","midpoint"],default="baseline"); p.add_argument("--batch",type=int,default=29)
    p.add_argument("--chunk",type=int,default=0); p.add_argument("--reuse",action="store_true"); p.add_argument("--fused",action="store_true"); p.add_argument("--checkpoint",action="store_true")
    p.add_argument("--loss-backend",choices=["torch","triton"],default="torch")
    p.add_argument("--swiglu",action="store_true")
    p.add_argument("--allocator",choices=["default","expandable"],default="default")
    p.add_argument("--warmup",type=int,default=5); p.add_argument("--updates",type=int,default=20)
    a=p.parse_args()
    for key in ("config","data_dir","output"): setattr(a,key,getattr(a,key).resolve())
    if a.batch<1 or a.chunk<0 or a.warmup<1 or a.updates<3: p.error("Positive batch, warmup, >=3 updates and nonnegative chunk required")
    if a.cpu_smoke and (a.loss_backend!="torch" or a.swiglu or a.allocator!="default"): p.error("CUDA kernels/allocator cannot be smoke-tested as CPU performance")
    if not a.chunk and not a.gate and (a.loss_backend!="torch" or a.swiglu): p.error("Kernel candidates require bounded head/loss path")
    if a.reuse and (a.method!="midpoint" or (not a.chunk and not a.gate)): p.error("Reuse requires midpoint chunked path")
    return a

if __name__=="__main__":
    args=parse_args()
    suite(args) if args.suite else worker(args)
