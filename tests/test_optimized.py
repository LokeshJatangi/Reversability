import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from reversibility import build_model
from reversibility.optimized import chunked_backward, cpu_snapshot, ReusedMidpoint
from reversibility.reversible import forward_pair
from benchmark_optimized import correctness_gate, optimizer_for, optimized_update, checkpoint_cpu
from test_baseline import _write_tiny_training_fixture

@pytest.mark.parametrize("method",["baseline","midpoint"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("chunk",[1,7,100])
def test_chunked_tied_weight_gradients_masks_accumulation(method,dtype,chunk):
    config={"model":{"vocab_size":31,"max_sequence_length":8,"n_layers":3,"n_heads":2,"d_model":8,"d_ff":16},"method":method}
    torch.manual_seed(23)
    original=build_model(config).to(dtype)
    candidate=copy.deepcopy(original)
    for micro in range(2):
        ids=torch.randint(31,(3,8)); labels=ids.roll(-1,1).clone(); labels[-1,:]=-100; labels[0,-1]=-100
        loss=F.cross_entropy(original(ids).flatten(0,1),labels.flatten(),ignore_index=-100,reduction="sum")
        (loss/91).backward()
        actual=chunked_backward(candidate,ids,labels,91,chunk,reuse=method=="midpoint")
        torch.testing.assert_close(actual,loss.detach().float(),rtol=2e-6,atol=2e-5)
    tolerance=2e-5 if dtype==torch.float32 else 1e-10
    for a,b in zip(original.parameters(),candidate.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=tolerance,atol=tolerance*.01)
    assert candidate.lm_head.weight is candidate.token_embedding.weight

@pytest.mark.parametrize("h",[.25,.5,.75])
def test_reused_midpoint_state_vjp_vs_unrolled(h):
    model=build_model({"model":{"vocab_size":31,"n_layers":3,"n_heads":2,"d_model":8,"d_ff":16},"method":"midpoint"}).double()
    copy_model=copy.deepcopy(model)
    p=torch.randn(2,4,8,dtype=torch.float64,requires_grad=True)
    q=torch.randn_like(p,requires_grad=True)
    left_p,left_q=p,q
    for block in model.blocks: left_p,left_q=forward_pair(block,left_p,left_q,"midpoint",h)
    right_p0=p.detach().clone().requires_grad_(True); right_q0=q.detach().clone().requires_grad_(True)
    right_p,right_q=ReusedMidpoint.apply(right_p0,right_q0,copy_model.blocks,h,*copy_model.blocks.parameters())
    dp=torch.randn_like(p); dq=torch.randn_like(q)
    (left_p*dp+left_q*dq).sum().backward(); (right_p*dp+right_q*dq).sum().backward()
    torch.testing.assert_close(left_p,right_p,rtol=0,atol=0); torch.testing.assert_close(left_q,right_q,rtol=0,atol=0)
    for a,b in [(p.grad,right_p0.grad),(q.grad,right_q0.grad),*[(a.grad,b.grad) for a,b in zip(model.blocks.parameters(),copy_model.blocks.parameters())]]:
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)


def test_cpu_checkpoint_independent_snapshot_and_exact_continuation(tmp_path):
    config=json.loads(_write_tiny_training_fixture(tmp_path,"fixture").read_text())
    config.update(method="midpoint",precision="fp32",gradient_accumulation_steps=1)
    model=build_model(config); optimizer=optimizer_for(model,config)
    train=np.memmap(Path(config["data_dir"])/"train.bin",dtype=np.uint16,mode="r")
    count=config["physical_batch_size"]*config["sequence_length"]*config["gradient_accumulation_steps"]
    optimized_update(model,optimizer,None,train,config,0,torch.device("cpu"),3,True)
    frozen=cpu_snapshot(model.state_dict())
    record=checkpoint_cpu(model,optimizer,None,config,count,"fixture",tmp_path,torch.device("cpu"))
    optimized_update(model,optimizer,None,train,config,count,torch.device("cpu"),3,True)
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
    checkpoint=torch.load(record["path"],weights_only=False,map_location="cpu")
    restored=build_model(config); restored.load_state_dict(checkpoint["model"])
    opt=optimizer_for(restored,config); opt.load_state_dict(checkpoint["optimizer"])
    optimized_update(restored,opt,None,train,config,checkpoint["committed_targets"],torch.device("cpu"),3,True)
    for k,v in model.state_dict().items(): torch.testing.assert_close(restored.state_dict()[k],v,rtol=0,atol=0)
    assert record["roundtrip_verified"]


def test_gate_predeclared_tolerances(tmp_path):
    config=json.loads(_write_tiny_training_fixture(tmp_path,"fixture").read_text())
    config["precision"]="fp32"
    result=correctness_gate(config,torch.device("cpu"))
    assert result["passed"]
    assert result["limits"]["gradient_relative_l2"]==5e-5


def test_worker_real_physical_50_without_accumulation(tmp_path, monkeypatch):
    from argparse import Namespace
    import hashlib
    import benchmark_optimized as benchmark
    config_path = _write_tiny_training_fixture(tmp_path, "fixture")
    config = json.loads(config_path.read_text())
    config["max_train_targets"] = 4096
    config_path.write_text(json.dumps(config))
    data = Path(config["data_dir"])
    (np.arange(4097) % 31).astype(np.uint16).tofile(data / "train.bin")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["training_targets"] = 4096
    manifest["files"]["train"]["sha256"] = hashlib.sha256((data / "train.bin").read_bytes()).hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    original = benchmark.chunked_backward
    def observe(model, ids, labels, denominator, *args, **kwargs):
        calls.append((tuple(ids.shape), int((labels != -100).sum()), denominator))
        return original(model, ids, labels, denominator, *args, **kwargs)
    monkeypatch.setattr(benchmark, "chunked_backward", observe)
    output = tmp_path / "physical50"
    benchmark.worker(Namespace(config=config_path, data_dir=data, output=output,
        method="midpoint", batch=50, chunk=64, reuse=True, fused=False, gate=False,
        checkpoint=False, cpu_smoke=True, warmup=1, updates=3, loss_backend="torch",swiglu=False,allocator="default"))
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["config"]["gradient_accumulation_steps"] == 1
    assert result["microstep_batch_sizes"] == [50]
    assert result["effective_batch_size"] == 50
    assert result["valid_targets_per_update"] == 200
    assert calls == [((50, 4), 200, 200)] * 4
    assert result["timing"]["measured_valid_targets"] == 600
    assert result["capacity"]["passed"] is None  # CPU proves shapes, not CUDA feasibility.


@pytest.mark.parametrize('override', [
    {'gradient_accumulation_steps': 2}, {'effective_batch_size': 100}])
def test_capacity_rejects_accumulated_or_mislabeled_batch(override):
    from benchmark_optimized import microstep_batches
    config = {'physical_batch_size': 50, 'gradient_accumulation_steps': 1,
              'effective_batch_size': 50, **override}
    with pytest.raises(ValueError, match='real physical batch'):
        microstep_batches(config)


@pytest.mark.parametrize('unexpected_failure', [False, True])
def test_capacity_boundary_confirmations_and_runtime_failure(tmp_path, monkeypatch, unexpected_failure):
    """Exercise the GPU orchestration with explicit simulated memory/runtime outcomes."""
    from argparse import Namespace
    import benchmark_optimized as benchmark
    calls = []
    def run(command, **kwargs):
        def arg(name): return command[command.index(name) + 1]
        output = Path(arg('--output')); output.mkdir()
        method = arg('--method'); batch = int(arg('--batch')); gate = '--gate' in command
        limit = 45 if method == 'baseline' else 53
        result = {'status': 'complete', 'gate': {'passing_chunks': [1024]},
            'config': {'method': method, 'physical_batch_size': batch, 'gradient_accumulation_steps': 1},
            'chunk_tokens': int(arg('--chunk')), 'force_reuse': '--reuse' in command,
            'fused_optimizer': '--fused' in command, 'loss_backend': arg('--loss-backend'),
            'fused_swiglu': '--swiglu' in command, 'allocator': arg('--allocator'),
            'capacity': {'passed': batch <= limit, 'peak_reserved_bytes': batch * 10**6},
            'timing': {'targets_per_second': 100, 'peak_reserved_bytes': batch * 10**6,
                       'peak_allocated_bytes': batch * 800000}, 'microstep_batch_sizes': [batch]}
        if not gate and batch > limit:
            result.update(status='failed', error='CUDA out of memory')
        if unexpected_failure and output.name == 'capacity-midpoint-b48':
            result.update(status='failed', error='AMP overflow; no speed claim')
        (output / 'result.json').write_text(json.dumps(result))
        calls.append(command)
        return Namespace(returncode=0 if result['status'] == 'complete' else 1)
    monkeypatch.setattr(benchmark.subprocess, 'run', run)
    output = tmp_path / 'suite'
    benchmark.suite(Namespace(output=output, config=tmp_path/'config.json',
        data_dir=tmp_path/'data', cpu_smoke=False, warmup=5, updates=20))
    result = json.loads((output/'suite.json').read_text())
    baseline = result['capacity_search']['baseline']
    assert baseline['largest_verified_physical_batch'] == 45
    assert baseline['first_failing_physical_bound'] == 46
    midpoint = result['capacity_search']['midpoint']
    if unexpected_failure:
        assert midpoint['largest_verified_physical_batch'] is None
        assert midpoint['first_failing_physical_bound'] is None
        assert midpoint['search_error']['error'].startswith('AMP overflow')
        assert not midpoint['ceiling_reached']
    else:
        assert midpoint['largest_verified_physical_batch'] == 53
        assert midpoint['first_failing_physical_bound'] == 54
    confirmations = [c for c in calls if 'capacity-baseline-b45-confirm-' in c[c.index('--output')+1]]
    assert len(confirmations) == 3
    assert all(c[c.index('--warmup')+1] == '5' and c[c.index('--updates')+1] == '20' for c in confirmations)
    assert '--checkpoint' in confirmations[2]
    # Fused optimizer candidates are never combined with ungated force reuse.
    assert all(not ('--fused' in c and '--reuse' in c) for c in calls)
