#!/usr/bin/env python3
"""Non-training reconstruction/gradient gates with immutable tolerance policy."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reversibility import ModelConfig, ReversibleLM
from reversibility.reversible import forward_pair, inverse_pair

# Optimizer policy v2: first Adam update is sensitive to sign near zero.
# Keep strict gradient checks and additionally bound the whole update L2 error.
OPTIMIZER_TOLERANCES = {'fp64': {'atol': 1e-9, 'rtol': 1e-8},
                        'fp32': {'atol': 5e-5, 'rtol': 3e-4},
                        'fp16': {'atol': 1.2e-3, 'rtol': 0.0}}
UPDATE_RELATIVE_L2 = {'fp64': 1e-7, 'fp32': .01, 'fp16': .05}

TOLERANCES = {
    'fp64': {'atol': 1e-9, 'rtol': 1e-8},
    'fp32': {'atol': 3e-5, 'rtol': 3e-4},
    'fp16': {'atol': 5e-3, 'rtol': 5e-2},
}


def check_case(method, precision, device, depth, length, width=32, heads=2, h=0.5, seed=1337):
    torch.manual_seed(seed)
    config = ModelConfig(vocab_size=101, max_sequence_length=length, n_layers=depth,
                         n_heads=heads, d_model=width, d_ff=4*width)
    dtype = torch.float64 if precision == 'fp64' else torch.float32
    inverse = ReversibleLM(config, method, h).to(device=device, dtype=dtype)
    reference = copy.deepcopy(inverse); reference.backward_mode = 'autograd'
    amp = (lambda: torch.autocast(device_type='cuda', dtype=torch.float16)) if precision == 'fp16' else contextlib.nullcontext
    errors = {}
    def compare(name, actual, expected):
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
        delta = (actual.detach()-expected.detach()).abs()
        errors[name] = {'max_abs': delta.max().item(),
                        'relative_to_max_reference': delta.max().item()/max(expected.detach().abs().max().item(),1e-12)}
        try:
            policy = OPTIMIZER_TOLERANCES if name.startswith('optimizer_') else TOLERANCES
            torch.testing.assert_close(actual, expected, **policy[precision], msg=name)
        except AssertionError:
            print('FAILED CHECK', method, precision, name, errors[name], flush=True)
            raise
    ids = torch.randint(0,config.vocab_size,(2,length),device=device)
    labels = torch.randint(0,config.vocab_size,(2,length),device=device)
    labels[-1,-1] = -100  # probe partial-target normalization
    with amp(), torch.no_grad():
        x = inverse.token_embedding(ids)+inverse.position_embedding(torch.arange(length,device=device))
        p, q = x,x
        states=[(p,q)]
        for block in inverse.blocks:
            p,q=forward_pair(block,p,q,method,h); states.append((p,q))
        for i in range(depth-1,-1,-1):
            p,q=inverse_pair(inverse.blocks[i],p,q,method,h)
            compare(f'state_{i}_p',p,states[i][0]); compare(f'state_{i}_q',q,states[i][1])
    # Independent equation expression for the stored-state reference.
    def reference_core(model,x):
        p,q=x,x
        for b in model.blocks:
            if method=='midpoint':
                a=b.attn(b.attn_norm(q)); force=a+b.mlp(b.mlp_norm(q+a))
                p,q=q,p+(2*h)*force
            else:
                q=q+b.attn(b.attn_norm(p)); p=p+b.mlp(b.mlp_norm(q))
        return q if method=='midpoint' else p
    with amp():
        x1=x.detach().requires_grad_(True); x2=x.detach().requires_grad_(True)
        out1=inverse.core(x1); out2=reference_core(reference,x2)
        compare('core_output',out1,out2)
        probe=torch.randn_like(out1)
        (out1*probe).mean().backward(); (out2*probe).mean().backward()
        compare('input_gradient',x1.grad,x2.grad)
    for name,p1 in inverse.named_parameters():
        p2=dict(reference.named_parameters())[name]
        if p1.grad is not None: compare('core_gradient_'+name,p1.grad,p2.grad)
    inverse.zero_grad(); reference.zero_grad()
    with amp():
        logits1=inverse(ids); logits2=reference(ids)
        compare('logits',logits1,logits2)
        loss1=F.cross_entropy(logits1.flatten(0,1),labels.flatten())
        loss2=F.cross_entropy(logits2.flatten(0,1),labels.flatten())
        compare('loss',loss1,loss2)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=6e-4,betas=(.9,.95),weight_decay=.1) for m in [inverse,reference]]
    scalers=[torch.amp.GradScaler('cuda') for _ in optimizers] if precision=='fp16' else None
    if scalers:
        for scaler,opt,loss in zip(scalers,optimizers,[loss1,loss2]):
            scaler.scale(loss).backward();scaler.unscale_(opt)
    else:
        loss1.backward();loss2.backward()
    for (name,p1),(name2,p2) in zip(inverse.named_parameters(),reference.named_parameters()):
        assert name==name2 and p1.grad is not None and p2.grad is not None
        compare('gradient_'+name,p1.grad,p2.grad)
    # AdamW's first update can amplify near-zero gradient differences; test it too.
    before = {name:p.detach().clone() for name,p in reference.named_parameters()}
    for i,(m,opt) in enumerate(zip([inverse,reference],optimizers)):
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        assert torch.isfinite(norm)
        if scalers:
            old=scalers[i].get_scale();scalers[i].step(opt);scalers[i].update()
            assert scalers[i].get_scale()>=old, 'overflow in optimizer gate'
        else:opt.step()
    for (name,p1),(_,p2) in zip(inverse.named_parameters(),reference.named_parameters()):
        compare('optimizer_'+name,p1,p2)
    diff_sq=sum((p1.detach()-p2.detach()).double().square().sum().item() for p1,p2 in zip(inverse.parameters(),reference.parameters()))
    update_sq=sum((p.detach()-before[name]).double().square().sum().item() for name,p in reference.named_parameters())
    relative_update=(diff_sq/max(update_sq,1e-30))**.5
    errors['optimizer_relative_update_l2'] = relative_update
    assert relative_update <= UPDATE_RELATIVE_L2[precision], ('optimizer update L2',relative_update)
    return {'method':method,'precision':precision,'depth':depth,'length':length,'width':width,
            'step_size': h if method=='midpoint' else None,'seed':seed,'passed':True,'errors':errors}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['midpoint','euler','both'],default='both')
    args=p.parse_args()
    precisions=['fp64','fp32'] if args.device=='cpu' else ['fp64','fp32','fp16']
    methods=['midpoint','euler'] if args.method=='both' else [args.method]
    report={'policy':'reconstruction-gradient-v2','tolerances':TOLERANCES,'optimizer_tolerances':OPTIMIZER_TOLERANCES,'optimizer_relative_l2_limits':UPDATE_RELATIVE_L2,'device':args.device,
            'torch':torch.__version__,'python':platform.python_version(),
            'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
            'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                             [Path(__file__),Path('src/reversibility/reversible.py'),Path('src/reversibility/model.py')]},
            'cases':[],'passed':False,'command':sys.argv}
    try:
        for method in methods:
            for precision in precisions:
                for depth,length in [(1,8),(3,17),(7,32)]:
                    for h in ([.25,.5] if method=='midpoint' else [.5]):
                        for seed in [1337,2026]:
                            report['cases'].append(check_case(method,precision,args.device,depth,length,h=h,seed=seed))
                # Architecture-sized, full-context probe (small vocab, no train targets).
                report['cases'].append(check_case(method,precision,args.device,7,512,width=256,heads=8))
        report['passed']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__': main()
