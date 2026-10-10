#!/usr/bin/env python3
"""Focused, feature-gated large-batch midpoint experiments. Probes, not full training."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
import torch
import benchmark_optimized as base
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from reversibility.optimized_v3 import install_v3,cached_head_backward,TiledMLP
from reversibility.optimized import amp_context

ORIGINAL_CHUNKED=base.chunked_backward
ORIGINAL_GATE=base.correctness_gate
FEATURE_KEYS=('core_mode','mlp_tokens','swiglu','rmsnorm','cached_head','allocator')


def features(args):
    return {key:getattr(args,key) for key in FEATURE_KEYS}


def tile_gate(config,device,args):
    """Actual configured MLP tile plus a tail; test gradient and two updates.

    Full-model gate B8/context512 may not cross8192-token MLP tiles. This
    independent probe crosses that boundary without allocating vocabulary logits.
    """
    seed=config['seed'];torch.manual_seed(seed)
    full=base.build_model(config).to(device)
    left=copy.deepcopy(full.blocks[0].mlp);right=copy.deepcopy(left)
    del full
    lr=config['learning_rate']
    a=torch.optim.AdamW(left.parameters(),lr=lr,betas=(config['beta1'],config['beta2']),weight_decay=config['weight_decay'])
    b=torch.optim.AdamW(right.parameters(),lr=lr,betas=(config['beta1'],config['beta2']),weight_decay=config['weight_decay'])
    amp=device.type=='cuda' and config['precision']=='fp16';scale=128. if amp else 1.
    limits={'gradient_relative_l2':.02 if amp else 5e-5,'loss_relative':.002 if amp else 2e-6,
            'optimizer_step_max_absolute':.0012 if amp else 5e-5,'optimizer_relative_update_l2':.05 if amp else .01}
    rows=args.mlp_tokens+17;records=[]
    for update in range(2):
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        x=torch.randn(rows,config['model']['d_model'],device=device,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        upstream=torch.randn_like(x)
        with amp_context(device,config['precision']):
            out=left(x)
            candidate=TiledMLP.apply(y,right.up.weight,right.down.weight,args.mlp_tokens,args.swiglu)
            loss=(out.float()*upstream).sum()/rows
            other=(candidate.float()*upstream).sum()/rows
        (loss*scale).backward();(other*scale).backward()
        pairs=[(x.grad/scale,y.grad/scale)]
        for p,q in zip(left.parameters(),right.parameters()):
            p.grad.div_(scale);q.grad.div_(scale);pairs.append((p.grad,q.grad))
        if any(not torch.isfinite(v).all() for pair in pairs for v in pair):raise FloatingPointError('Nonfinite tiled MLP gate')
        grad=max(float((u-v).norm()/u.norm().clamp_min(1e-12)) for u,v in pairs)
        torch.nn.utils.clip_grad_norm_(left.parameters(),config['grad_clip']);torch.nn.utils.clip_grad_norm_(right.parameters(),config['grad_clip'])
        old=[p.detach().clone() for p in left.parameters()];old_right=[p.detach().clone() for p in right.parameters()]
        a.step();b.step()
        diff=sum((p-q).detach().double().square().sum().item() for p,q in zip(left.parameters(),right.parameters()))
        increment=sum((p.detach()-u).double().square().sum().item() for p,u in zip(left.parameters(),old))
        row={'update':update,'tokens':rows,'configured_tile':args.mlp_tokens,'gradient_relative_l2':grad,
             'loss_relative':float((loss-other).detach().abs()/loss.detach().abs().clamp_min(1e-12)),
             'optimizer_step_max_absolute':max(float(((p.detach()-u)-(q.detach()-v)).abs().max()) for p,q,u,v in zip(left.parameters(),right.parameters(),old,old_right)),
             'optimizer_relative_update_l2':math.sqrt(diff/max(increment,1e-30))}
        row['passed']=all(row[key]<=value for key,value in limits.items());records.append(row)
    return {'limits':limits,'records':records,'passed':all(row['passed'] for row in records),'probe_scale':scale}


def worker(args):
    declared=features(args)
    prior_headroom=base.HEADROOM;base.HEADROOM=args.headroom
    def backward(model,ids,labels,denominator,chunk,scaler=None,precision='fp32',reuse=False,loss_backend='torch'):
        mode=args.core_mode if getattr(model,'method',None)=='midpoint' else 'reference'
        install_v3(model,core_mode=mode,mlp_tokens=args.mlp_tokens,swiglu=args.swiglu,rmsnorm=args.rmsnorm)
        function=cached_head_backward if args.cached_head else ORIGINAL_CHUNKED
        return function(model,ids,labels,denominator,chunk,scaler,precision,False,loss_backend)
    def gate(config,device,fused=False,loss_backend='torch',swiglu=False,reuse=False):
        result=ORIGINAL_GATE(config,device,fused,loss_backend,swiglu,False)
        result['v3_features']=declared
        result['core_mode_by_method']={'baseline':'reference','midpoint':args.core_mode}
        if args.mlp_tokens:
            result['tiled_mlp_boundary']=tile_gate(config,device,args)
            if not result['tiled_mlp_boundary']['passed']:
                result['passed']=False;result['passing_chunks']=[]
        return result
    base.chunked_backward=backward;base.correctness_gate=gate
    args.reuse=False
    try:
        base.worker(args)
    finally:
        path=args.output/'result.json'
        if path.exists():
            result=json.loads(path.read_text());result['v3_features']=declared
            result['actual_core_mode']=args.core_mode if args.method=='midpoint' else 'reference'
            result['v3_harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            base.save(path,result)
        base.chunked_backward=ORIGINAL_CHUNKED;base.correctness_gate=ORIGINAL_GATE;base.HEADROOM=prior_headroom


def candidate_paths(cpu=False):
    default={'core_mode':'reference','mlp_tokens':0,'swiglu':False,'rmsnorm':False,'cached_head':False,'allocator':'default'}
    choices=[('anchor',{}),('reuse-cached',{'core_mode':'reuse','swiglu':not cpu,'cached_head':True}),
        ('split-cached',{'core_mode':'split','swiglu':not cpu,'cached_head':True}),
        ('split-tiled',{'core_mode':'split','mlp_tokens':37 if cpu else 8192,'swiglu':not cpu,'cached_head':True})]
    if not cpu:
        choices += [('split-tiled-norm',{'core_mode':'split','mlp_tokens':8192,'swiglu':True,'rmsnorm':True,'cached_head':True}),
                    ('split-tiled-norm-expand',{'core_mode':'split','mlp_tokens':8192,'swiglu':True,'rmsnorm':True,'cached_head':True,'allocator':'expandable'})]
    return [(name,{**default,**changes}) for name,changes in choices]


def suite(args):
    args.output.mkdir(parents=True,exist_ok=False);records=[];searches={};references=[]
    def persist(extra=None):
        base.save(args.output/'suite.json',{'records':records,'searches':searches,'headroom':args.headroom,'accumulation':1,'cpu_smoke_only':args.cpu_smoke,'validation_loss_ceiling':5.56229485,**(extra or {})})
    def invoke(name,choice,method='midpoint',batch=814,chunk=1024,gate=False,short=False,checkpoint=False,loss_backend=None):
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config),'--data-dir',str(args.data_dir),
            '--output',str(args.output/name),'--method',method,'--batch',str(batch),'--chunk',str(chunk),
            '--warmup',str(1 if args.cpu_smoke else 3 if short else args.warmup),
            '--updates',str(3 if args.cpu_smoke or short else args.updates),'--core-mode',choice['core_mode'],
            '--mlp-tokens',str(choice['mlp_tokens']),'--allocator',choice['allocator'],
            '--loss-backend',loss_backend or ('torch' if args.cpu_smoke else 'triton'),'--headroom',str(args.headroom)]
        for flag,on in [('--swiglu',choice['swiglu']),('--rmsnorm',choice['rmsnorm']),('--cached-head',choice['cached_head']),
                        ('--gate',gate),('--checkpoint',checkpoint),('--cpu-smoke',args.cpu_smoke)]:
            if on:command.append(flag)
        environment=os.environ.copy();environment.pop('PYTORCH_ALLOC_CONF',None);environment.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        if choice['allocator']=='expandable':environment['PYTORCH_ALLOC_CONF']='expandable_segments:True'
        print('RUN',name,flush=True)
        with (args.output/(name+'.log')).open('w') as log:
            process=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        path=args.output/name/'result.json'
        row={'name':name,'returncode':process.returncode,'result':json.loads(path.read_text()) if path.exists() else {'status':'failed','error':'Missing worker result'}}
        records.append(row);persist();print('RESULT',name,row['result']['status'],flush=True)
        return row
    def verdict(row):
        v=row['result']
        if v['status']=='complete':return 'fit' if v.get('capacity',{}).get('passed') else 'headroom'
        return 'oom' if 'out of memory' in v.get('error','').lower() else 'runtime_failure'
    def search(label,choice,chunk,method='midpoint'):
        lower=0;upper=None;error=None;fit_result=None;trials=[]
        # Rank each candidate AT large batch, rather than choosing from batch50 memory.
        shape_grid=(50,768,814,896,1024,1152,1280,1536,1792,2048) if method=='midpoint' else (50,256,285,384,512,768,1024,1536,2048)
        for batch in shape_grid:
            row=invoke(f'capacity-{label}-b{batch}',choice,method,batch,chunk,short=True);trials.append(row['name']);outcome=verdict(row)
            if outcome=='fit':lower=batch;fit_result=row['result']
            elif outcome=='runtime_failure':error={'trial':row['name'],'error':row['result'].get('error')};break
            else:upper=batch;break
        if upper is not None and error is None:
            while upper-lower>1:
                batch=(upper+lower)//2
                row=invoke(f'capacity-{label}-b{batch}-refine',choice,method,batch,chunk,short=True);trials.append(row['name']);outcome=verdict(row)
                if outcome=='fit':lower=batch;fit_result=row['result']
                elif outcome=='runtime_failure':error={'trial':row['name'],'error':row['result'].get('error')};break
                else:upper=batch
        data={'discovery_physical_batch':lower if error is None else None,'first_failing_bound':upper,'runtime_failure':error,
              'search_ceiling':2048,'ceiling_reached':lower==2048 and error is None,'features':choice,'chunk_tokens':chunk,
              'fit_peak_reserved_bytes':None if fit_result is None else fit_result['capacity']['peak_reserved_bytes'],
              'largest_verified_physical_batch':None,'trials':trials}
        searches[label]=data;persist();return data
    choices=candidate_paths(args.cpu_smoke);anchor=choices[0][1];passing=[]
    if args.variant_set!='all':
        choices=[(name,choice) for name,choice in choices if name==args.variant_set]
        if not choices:raise ValueError('Variant unavailable in this runtime')
    for label,choice in choices:
        row=invoke('gate-'+label,choice,batch=2 if args.cpu_smoke else 29,gate=True)
        g=row['result'].get('gate',{})
        if row['result']['status']=='complete' and g.get('passed'):
            chunk=3 if args.cpu_smoke else 1024 if 1024 in g['passing_chunks'] else min(g['passing_chunks'])
            passing.append((label,choice,chunk))
    if not passing:raise RuntimeError('No v3 configuration passed; evidence preserved')
    for i in range(3):
        row=invoke('reference-start-'+str(i),anchor,'baseline',2 if args.cpu_smoke else 29,0,loss_backend='torch');references.append(row)
    if args.cpu_smoke:
        for label,choice,chunk in passing:
            for batch in (2,3):invoke(f'cpu-{label}-b{batch}',choice,batch=batch,chunk=chunk,checkpoint=batch==3)
        selected=None
    else:
        if args.capacity_search:
            for label,choice,chunk in passing:
                invoke('matched50-midpoint-'+label,choice,batch=50,chunk=chunk)
                invoke('matched50-baseline-'+label,choice,method='baseline',batch=50,chunk=chunk)
                search(label,choice,chunk)
            feasible=[(label,data) for label,data in searches.items() if data['discovery_physical_batch']]
            selected=None
            if feasible:
                label,data=max(feasible,key=lambda item:(item[1]['discovery_physical_batch'],-item[1]['fit_peak_reserved_bytes']))
                confirmations=[invoke(f'confirm-{label}-{i}',data['features'],batch=data['discovery_physical_batch'],chunk=data['chunk_tokens'],checkpoint=i==2) for i in range(3)]
                if all(verdict(row)=='fit' for row in confirmations):
                    data['largest_verified_physical_batch']=data['discovery_physical_batch'];selected={'label':label,**data}
                data['confirmations']=[row['name'] for row in confirmations];persist()
                control=search('matched-baseline',data['features'],data['chunk_tokens'],method='baseline')
                if control['discovery_physical_batch']:
                    confirmations=[invoke(f'confirm-matched-baseline-{i}',control['features'],method='baseline',batch=control['discovery_physical_batch'],chunk=control['chunk_tokens'],checkpoint=i==2) for i in range(3)]
                    if all(verdict(row)=='fit' for row in confirmations):control['largest_verified_physical_batch']=control['discovery_physical_batch']
                    control['confirmations']=[row['name'] for row in confirmations];persist()
        else:
            # Optimization review precedes any maximum-search decision.
            large_results=[]
            grid=(768,814,832,1024) if args.headroom==.10 else (814,832,896,1024)
            for label,choice,chunk in passing:
                invoke('matched50-midpoint-'+label,choice,batch=50,chunk=chunk)
                invoke('matched50-baseline-'+label,choice,method='baseline',batch=50,chunk=chunk)
                for batch in grid:
                    row=invoke(f'probe-{label}-b{batch}',choice,batch=batch,chunk=chunk,short=True)
                    if verdict(row)=='fit':large_results.append((label,choice,chunk,row))
                    if verdict(row)=='runtime_failure':break
            # Fixed814 comparison at10%; explore higher shapes separately at5%.
            fixed=[item for item in large_results if item[3]['result']['config']['physical_batch_size']==814]
            candidates=fixed if args.headroom==.10 else large_results
            selected=None
            if candidates:
                label,choice,chunk,row=min(candidates,key=lambda item:(
                    -item[3]['result']['config']['physical_batch_size'] if args.headroom==.05 else 0,
                    item[3]['result']['capacity']['peak_reserved_bytes']))
                batch=row['result']['config']['physical_batch_size']
                confirmations=[invoke(f'confirm-{label}-b{batch}-{i}',choice,batch=batch,chunk=chunk,checkpoint=i==2) for i in range(3)]
                verified=all(verdict(row)=='fit' for row in confirmations)
                selected={'label':label,'features':choice,'chunk_tokens':chunk,'tested_physical_batch':batch,
                    'verified_probe_batch':batch if verified else None,'confirmations':[row['name'] for row in confirmations],
                    'absolute_maximum_established':False,'headroom':args.headroom}
                invoke('shared-baseline285-'+label,choice,method='baseline',batch=285,chunk=chunk)
                # Repeated814 anchor makes its comparison independent of old-session drift.
                if args.headroom==.10 and label!='anchor' and any(name=='anchor' for name,_,_ in passing):
                    for i in range(3):invoke(f'anchor814-repeat-{i}',anchor,batch=814,chunk=1024,checkpoint=i==2)
    for i in range(3):
        row=invoke('reference-end-'+str(i),anchor,'baseline',2 if args.cpu_smoke else 29,0,loss_backend='torch');references.append(row)
    rates=[row['result']['timing']['targets_per_second'] for row in references if row['result']['status']=='complete']
    extra={'status':'cpu_smoke_complete' if args.cpu_smoke else 'experiments_complete','selected_midpoint':selected,
           'prior_verified_midpoint_physical_batch':814,'capacity_search_requested':args.capacity_search,'baseline29_reference_targets_per_second':rates,
           'note':'One real physical batch per update; memory-first ranking from large-batch trials. Default mode reviews optimization probes, not an absolute maximum; 5% headroom is a separate exploratory protocol. Discovery fits are provisional; only selected paths receiving three full-duration confirmations including CPU checkpoint are verified. Internal token-independent MLP tiling is disclosed and shared with baseline control; attention processes the full batch. No full-run loss/savings claim. Runtime failures are not memory boundaries.'}
    persist(extra)
    lines=['# Midpoint v3 experiments','',extra['note'],'','| Probe | Status | Physical batch | Targets/s | Peak allocated GiB | Peak reserved GiB | Fits memory budget |','|---|---|---:|---:|---:|---:|---|']
    for row in records:
        v=row['result'];cap=v.get('capacity',{});timing=v.get('timing',{})
        lines.append('| '+ ' | '.join([row['name'],v['status'],str(v.get('config',{}).get('physical_batch_size','-')),
            str(round(timing['targets_per_second'],1)) if 'targets_per_second' in timing else '-',
            str(round(cap['peak_allocated_bytes']/2**30,3)) if 'peak_allocated_bytes' in cap else '-',
            str(round(cap['peak_reserved_bytes']/2**30,3)) if 'peak_reserved_bytes' in cap else '-',str(cap.get('passed','-'))])+' |')
    lines+=['','```json',json.dumps({'selected_midpoint':selected,'searches':searches},indent=2),'```']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('suite','gate','cpu-smoke','checkpoint','swiglu','rmsnorm','cached-head','fused'):p.add_argument('--'+name,action='store_true')
    p.add_argument('--headroom',type=float,choices=[.10,.05],default=.10)
    p.add_argument('--capacity-search',action='store_true',help='Deferred: require a separate maximum-search decision')
    p.add_argument('--variant-set',default='all',choices=['all','anchor','reuse-cached','split-cached','split-tiled','split-tiled-norm','split-tiled-norm-expand'])
    p.add_argument('--method',choices=['baseline','midpoint'],default='midpoint')
    p.add_argument('--batch',type=int,default=814);p.add_argument('--chunk',type=int,default=1024)
    p.add_argument('--core-mode',choices=['reference','reuse','split'],default='reference')
    p.add_argument('--mlp-tokens',type=int,default=0)
    p.add_argument('--allocator',choices=['default','expandable'],default='default')
    p.add_argument('--loss-backend',choices=['torch','triton'],default='triton')
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--updates',type=int,default=20)
    args=p.parse_args()
    for name in ('config','data_dir','output'):setattr(args,name,getattr(args,name).resolve())
    if args.batch<1 or args.chunk<0 or args.mlp_tokens<0 or args.warmup<1 or args.updates<3:p.error('Positive batch/warmup, >=3 updates, nonnegative chunk/tile required')
    if args.cpu_smoke and (args.swiglu or args.rmsnorm or args.loss_backend!='torch' or args.allocator!='default'):p.error('CPU smoke cannot execute CUDA kernels/allocator')
    if not args.chunk and not args.gate and any((args.cached_head,args.mlp_tokens,args.swiglu,args.rmsnorm,args.core_mode!='reference')):p.error('v3 features require chunked path')
    return args

if __name__=='__main__':
    args=parse_args()
    suite(args) if args.suite else worker(args)
