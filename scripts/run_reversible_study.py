#!/usr/bin/env python3
"""Drive-persistent staged study; matched runs precede explicit review and max run."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
NAMES={m:f'{m}_20m_fineweb_edu_50m_v1_matched' for m in ['midpoint','euler']}
LOSS_DELTA=0.10  # predeclared absolute nats/target, frozen before matched runs


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):d.update(b)
    return d.hexdigest()


def freeze(path,value):
    text=json.dumps(value,indent=2)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:
        raise RuntimeError(f'frozen artifact differs: {path}; preserve it and investigate drift')
    if not path.exists():path.write_text(text)
    return path


def logged(command,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as log:
        header=f'\n[{datetime.now(timezone.utc).isoformat()}] $ {shlex.join(map(str,command))}\n'
        print(header,flush=True);log.write(header);log.flush()
        process=subprocess.Popen(list(map(str,command)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait();os.fsync(log.fileno())
    if code:raise subprocess.CalledProcessError(code,command)


def startup(run):
    return next(json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines() if json.loads(l)['event']=='startup')


def completed(run,budget=50_000_000):
    s=json.loads((run/'run_summary.json').read_text())
    if not s['completed_full_target_budget'] or s['committed_targets']!=budget or s['target_budget']!=budget:
        raise RuntimeError(f'incomplete run: {run}')
    if not math.isfinite(s['final_validation_loss']):raise RuntimeError(f'nonfinite validation: {run}')
    c=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    if c['committed_targets']!=budget or c['step']!=s['optimizer_steps']:raise RuntimeError('checkpoint/summary disagreement')
    return s


def train(config_path,budget=50_000_000):
    cfg=json.loads(config_path.read_text());run=Path(cfg['output_dir']);run.mkdir(parents=True,exist_ok=True)
    if (run/'run_summary.json').exists():
        s=json.loads((run/'run_summary.json').read_text())
        if s.get('completed_full_target_budget') and s.get('committed_targets')==budget:return completed(run,budget)
    command=[sys.executable,'-u',ROOT/'scripts/train_baseline.py','--config',config_path,'--device','cuda']
    if budget!=50_000_000:command+=['--max-targets',str(budget)]
    if (run/'latest.pt').exists():command+=['--resume',run/'latest.pt']
    logged(command,run/'console.log')
    return completed(run,budget)


def source_snapshot(study):
    paths=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'tests').glob('*.py')),
           ROOT/'notebooks/reversible_colab.ipynb',
           *sorted((ROOT/'configs').glob('*.json')),
           ROOT/'requirements.txt',ROOT/'AGENTS.md',
           ROOT/'docs/experiment-plan.md',ROOT/'docs/reversible-methods.md']
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    frozen=freeze(study/'source-manifest.json',manifest)
    digest=sha(frozen);snapshot=study/'source-snapshots'/f'{digest}.zip';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,p.relative_to(ROOT))
    return digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['matched','review','maximum'])
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('/content/reversibility-data/data_fineweb_edu_gpt2_50m_v1'))
    p.add_argument('--max-batch',type=int,default=256)
    args=p.parse_args();artifacts=args.artifacts.resolve();study=artifacts/'reversible-v1'
    baseline_run=artifacts/'runs/baseline_20m_fineweb_edu_50m_v1'
    baseline=completed(baseline_run);base=json.loads((artifacts/'configs/baseline_colab.json').read_text())
    baseline_start=startup(baseline_run)
    if sha(artifacts/'configs/baseline_colab.json')!=baseline_start['config_sha256']:raise RuntimeError('baseline config drift')
    policy={'validation_loss_max_delta_nats':LOSS_DELTA,'baseline_loss':baseline['final_validation_loss'],
            'selection_order':['correctness','loss_acceptance','lowest_final_loss','highest_end_to_end_throughput','lowest_peak_allocated_memory'],
            'physical_batch':base['physical_batch_size'],'accumulation':base['gradient_accumulation_steps'],
            'effective_batch':base['physical_batch_size']*base['gradient_accumulation_steps'],
            'target_budget':50_000_000,'midpoint_h':.5,'boundary_states':'both equal input embedding'}
    freeze(study/'selection-policy.json',policy)
    source_hash=source_snapshot(study)
    matched_configs={}
    for method in NAMES:
        cfg=dict(base)
        cfg.update(experiment_name=NAMES[method],method=method,step_size=.5,backward_mode='reconstruct',
                   data_dir=str(args.data_dir.resolve()),output_dir=str(study/'runs'/NAMES[method]),
                   source_manifest_sha256=source_hash)
        matched_configs[method]=freeze(study/'configs'/f'{method}_matched.json',cfg)
    if args.phase=='review':
        records={m:completed(study/'runs'/NAMES[m]) for m in NAMES}
        gpu_gate=json.loads((study/'correctness/cuda.json').read_text())
        if not gpu_gate['passed']:raise RuntimeError('CUDA correctness gate failed')
        acceptable=[m for m,s in records.items() if s['final_validation_loss']<=baseline['final_validation_loss']+LOSS_DELTA]
        selected=min(acceptable,key=lambda m:(records[m]['final_validation_loss'],-records[m]['end_to_end_targets_per_second_this_process'],records[m]['peak_cuda_allocated_bytes'])) if acceptable else None
        review={'selected_method':selected,'acceptable_methods':acceptable,'policy':policy,'baseline':baseline,'matched_runs':records,
                'planning_session_recorded':False,'review_notes':'Record correctness, trajectories, stability, failures and remaining Colab budget before stage 5.',
                'remaining_colab_budget':'REQUIRED: record remaining budget/availability',
                'protocol_amendments':[]}
        freeze(study/'review-proposal.json',review)
        print(json.dumps(review,indent=2));print('Hold the required review; save review-decision.json with planning_session_recorded=true, review_notes and remaining_colab_budget. No max search has run.')
        return
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; run on baseline Tesla T4 in Colab')
    expected=baseline_start['environment']
    actual={'torch':torch.__version__,'numpy':__import__('numpy').__version__,'cuda_runtime':torch.version.cuda}
    if torch.cuda.get_device_name()!=expected['accelerator']['name']:raise RuntimeError('baseline GPU type mismatch')
    for k,v in actual.items():
        if v!=expected[k]:raise RuntimeError(f'baseline software mismatch: {k}: {v} != {expected[k]}')
    if platform.python_version()!=expected['python'].split()[0]:raise RuntimeError('baseline Python version mismatch')
    if sha(args.data_dir/'manifest.json')!=baseline_start['manifest_sha256']:raise RuntimeError('use the exact Colab baseline dataset manifest')
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    for name,record in manifest['tokenizer']['files'].items():
        asset=args.data_dir/'tokenizer'/name
        if asset.stat().st_size!=record['bytes'] or sha(asset)!=record['sha256']:raise RuntimeError('tokenizer hash mismatch')
    for name in ['train','validation']:
        if sha(args.data_dir/f'{name}.bin')!=manifest['files'][name]['sha256']:raise RuntimeError('data hash mismatch')
    gate=study/'correctness/cuda.json'
    if not gate.exists():logged([sys.executable,ROOT/'scripts/validate_reversible.py','--device','cuda','--output',gate],study/'correctness/cuda.log')
    result=json.loads(gate.read_text())
    if not result['passed'] or result['gpu']!=torch.cuda.get_device_name() or result['torch']!=torch.__version__:raise RuntimeError('invalid CUDA correctness evidence')
    for name,digest in result['source_sha256'].items():
        if sha(Path(name))!=digest:raise RuntimeError('correctness source drift')
    if args.phase=='matched':
        for method,path in matched_configs.items():
            cfg=json.loads(path.read_text());smoke=dict(cfg)
            smoke.update(experiment_name=NAMES[method]+'_smoke',output_dir=str(study/'runs'/(NAMES[method]+'_smoke')),eval_interval=2,checkpoint_interval=2)
            smoke_path=freeze(study/'configs'/f'{method}_smoke.json',smoke)
            train(smoke_path,65_536);train(path)
        print('Matched runs complete. Run review phase and hold the planning session before maximum phase.')
        return
    # Maximum phase is explicitly blocked until both matched runs and review exist.
    records={m:completed(study/'runs'/NAMES[m]) for m in NAMES}
    decision=json.loads((study/'review-decision.json').read_text());proposal=json.loads((study/'review-proposal.json').read_text())
    if not decision.get('planning_session_recorded') or not decision.get('review_notes') or not decision.get('remaining_colab_budget') or decision['remaining_colab_budget'].startswith('REQUIRED'):
        raise RuntimeError('planning checkpoint is incomplete')
    method=decision['selected_method']
    if method is None:raise RuntimeError('no reversible variant passes loss acceptance; report this outcome')
    if method!=proposal['selected_method'] or decision['policy']!=policy:raise RuntimeError('selection differs from proposal/policy; document and review amendment first')
    cfg=json.loads(matched_configs[method].read_text())
    # Independent largest-batch run starts from seed 1337, never matched checkpoint.
    capacity=study/'benchmarks'/f'{method}_maximum_capacity.json'
    if capacity.exists():
        existing=json.loads(capacity.read_text())
        if existing.get('search_was_capped') and args.max_batch>existing['max_batch_tested']:
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            capacity.rename(capacity.with_name(capacity.stem+'-capped-'+stamp+'.json'))
    if not capacity.exists():logged([sys.executable,ROOT/'scripts/benchmark_batch_capacity.py','--config',matched_configs[method],
          '--output',capacity,'--max-batch',str(args.max_batch),'--repetitions','3','--headroom-fraction','.10'],capacity.with_suffix('.log'))
    cap=json.loads(capacity.read_text())
    if cap['search_was_capped']:raise RuntimeError('capacity search hit its ceiling; archive report and rerun with higher --max-batch before claiming a maximum')
    if cap['config_sha256']!=sha(matched_configs[method]) or cap['gpu_name']!=torch.cuda.get_device_name():raise RuntimeError('capacity/config mismatch')
    physical=cap['selected_physical_batch_size'];effective=policy['effective_batch']
    accumulation=effective//physical if effective%physical==0 else max(1,round(effective/physical))
    cfg.update(experiment_name=f'{method}_20m_fineweb_edu_50m_v1_maximum',physical_batch_size=physical,
               gradient_accumulation_steps=accumulation,capacity_search_report=str(capacity),
               output_dir=str(study/'runs'/f'{method}_20m_fineweb_edu_50m_v1_maximum'))
    max_cfg=freeze(study/'configs'/f'{method}_maximum.json',cfg)
    freeze(study/'maximum-batch-accounting.json',{'physical_batch':physical,'accumulation':accumulation,
       'baseline_effective_batch':effective,'maximum_effective_batch':physical*accumulation,
       'optimization_confound':physical*accumulation!=effective,
       'expected_optimizer_updates':math.ceil(50_000_000/(physical*accumulation*cfg['sequence_length']))})
    # Reconfirm chosen physical size with the actual maximum-run accumulation.
    verify=study/'benchmarks'/f'{method}_maximum_confirmation.json'
    if not verify.exists():logged([sys.executable,ROOT/'scripts/benchmark_batch_capacity.py','--config',max_cfg,
        '--candidate',str(physical),'--result-path',verify,'--repetitions','3','--headroom-fraction','.10'],verify.with_suffix('.log'))
    if json.loads(verify.read_text())['status']!='pass':raise RuntimeError('actual maximum-run accumulation failed capacity confirmation')
    train(max_cfg)

if __name__=='__main__': main()
