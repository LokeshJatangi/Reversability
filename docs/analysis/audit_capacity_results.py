from pathlib import Path
import hashlib,zipfile,json,statistics,math
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'20261010T154001Z-2382516a-capacity-v2.zip'
OUT=ROOT/'docs/analysis'
sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(ARCHIVE) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))
 prefix='20261010T154001Z-2382516a'
 suite=json.loads(z.read(prefix+'/suite.json'));receipt=json.loads(z.read(prefix+'-receipts.json'))
 members=[{'path':n,'bytes':len(z.read(n)),'sha256':sha(z.read(n))} for n in z.namelist()]
 by={r['name']:r['result'] for r in suite['records']}
 for name,v in by.items():assert json.loads(z.read(prefix+'/'+name+'/result.json'))==v
 for name,v in by.items():
  for p,h in receipt['embedded_sha256'].items():assert v['source_sha256'][p]==h
  assert v['config_file_sha256']==sha((ROOT/'results/recorded/configs/baseline_colab.json').read_bytes())
  assert v['config']['gradient_accumulation_steps']==1
  assert v['config']['precision']=='fp16'
  if 'gate' not in v and v['status']=='complete':
   b=v['config']['physical_batch_size'];ctx=v['config']['sequence_length'];n=len(v['timing']['update_seconds']);warmup=int(v['command'][v['command'].index('--warmup')+1])
   assert v['parameters']==20340736 and ctx==512
   assert v['effective_batch_size']==b and v['microstep_batch_sizes']==[b]
   assert v['valid_targets_per_update']==b*ctx
   assert v['timing']['measured_valid_targets']==b*ctx*n
   assert v['ordered_target_range']==[0,(n+warmup)*b*ctx]
   assert v['manifest_sha256']==receipt['data']['manifest_sha256']
   reserve=max(v['startup_memory']['peak_reserved_bytes'],v['timing']['peak_reserved_bytes'],v.get('checkpoint',{}).get('peak_reserved_bytes',0))
   allocated=max(v['startup_memory']['peak_allocated_bytes'],v['timing']['peak_allocated_bytes'],v.get('checkpoint',{}).get('peak_allocated_bytes',0))
   assert reserve==v['capacity']['peak_reserved_bytes'] and allocated==v['capacity']['peak_allocated_bytes']
   assert v['capacity']['passed']==(reserve<=.9*v['capacity']['total_device_bytes'])
   assert math.isclose(v['timing']['targets_per_second'],b*ctx*n/sum(v['timing']['update_seconds']),rel_tol=1e-12)
  if 'checkpoint' in v:
   assert v['checkpoint']['roundtrip_verified'] and not v['checkpoint']['included_in_training_timing']
   assert json.loads(z.read(prefix+'/'+name+'/checkpoint-overhead.json'))==v['checkpoint']
 gate_summary={}
 for name,v in by.items():
  if 'gate' not in v:continue
  g=v['gate'];assert v['status']=='complete' and g['passed']
  assert g['passing_chunks']==[1024,2048,4096] and len(g['records'])==12
  assert all(r['passed'] and all(r[k]<=limit for k,limit in g['limits'].items()) for r in g['records'])
  gate_summary[name]={'limits':g['limits'],'worst':{k:max(r[k] for r in g['records']) for k in g['limits']},'records':g['records'],'passed':True}
 for method,search in suite['capacity_search'].items():
  b=search['largest_verified_physical_batch'];assert search['first_failing_physical_bound']==b+1 and not search['ceiling_reached'] and search['search_error'] is None
  for repeat in range(3):
   v=by[f'capacity-{method}-b{b}-confirm-{repeat}'];assert v['status']=='complete' and v['capacity']['passed']
   assert len(v['timing']['update_seconds'])==20 and v['command'][v['command'].index('--warmup')+1]=='5'
   for k,value in search['selected_optimization'].items():assert v[k]==value
  assert 'checkpoint' in by[f'capacity-{method}-b{b}-confirm-2']
  fail=by[f'capacity-{method}-b{b+1}-refine'];assert fail['status']=='complete' and not fail['capacity']['passed']
 frozen=json.loads((ROOT/'results/recorded/reversible-v1/source-manifest.json').read_text())
 assert all(sha((ROOT/p).read_bytes())==h for p,h in frozen.items())
 assert all(v['source_sha256'][p]==h for v in by.values() for p,h in frozen.items() if p in v['source_sha256'])
 import ast
 notebook=json.loads((ROOT/'notebooks/capacity_colab_v2.ipynb').read_text())
 embedded=next(ast.literal_eval(n.value) for c in notebook['cells'] if c['cell_type']=='code' for n in ast.parse(''.join(c['source'])).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EMBEDDED' for t in n.targets))
 assert receipt['embedded_sha256']=={p:v['sha256'] for p,v in embedded.items()}
 def summarize(v):
  return {k:v[k] for k in ('status','parameters','capacity','checkpoint','chunk_tokens','force_reuse','fused_optimizer','loss_backend','fused_swiglu','allocator') if k in v} | {
    'physical_batch':v['config']['physical_batch_size'],'accumulation':v['config']['gradient_accumulation_steps'],
    'targets_per_second':v.get('timing',{}).get('targets_per_second'),'losses':v.get('timing',{}).get('losses'),
    'mean_update_seconds':v.get('timing',{}).get('mean_update_seconds'),
    'mean_gpu_busy_percent':v.get('gpu_utilization',{}).get('mean_gpu_busy_percent')}
 selected_names=[n for n in by if n.startswith(('final-','reference-','kernel-','allocator-')) and 'gate' not in n]+[n for n in by if '-confirm-' in n or n.endswith(('b286-refine','b815-refine'))]
 condensed={n:summarize(by[n]) for n in selected_names}
 totals={'records':len(by),'complete':sum(v['status']=='complete' for v in by.values()),'oom':sum(v['status']=='failed' for v in by.values()),'headroom_rejected':sum(v.get('capacity',{}).get('passed') is False for v in by.values())}
 max_summary={}
 for method,search in suite['capacity_search'].items():
  b=search['largest_verified_physical_batch'];vs=[by[f'capacity-{method}-b{b}-confirm-{i}'] for i in range(3)]
  max_summary[method]={'physical_batch':b,'next_batch_headroom_rejected':b+1,
    'median_targets_per_second':statistics.median(v['timing']['targets_per_second'] for v in vs),
    'peak_allocated_gib':max(v['capacity']['peak_allocated_bytes'] for v in vs)/2**30,
    'peak_reserved_gib':max(v['capacity']['peak_reserved_bytes'] for v in vs)/2**30,
    'reserved_percent':max(v['capacity']['reserved_fraction'] for v in vs)*100,
    'extra_room_before_90_percent_mib':min(v['capacity']['unused_reserved_budget_bytes'] for v in vs)/2**20,
    'checkpoint':vs[2]['checkpoint'],'updates_for_50m':math.ceil(50_000_000/(b*512)),
    'selected_optimization':search['selected_optimization']}
 a,b=by['kernel-baseline-b50-ce'],by['kernel-midpoint-b50-ce']
 assert all(a[k]==b[k] for k in ('chunk_tokens','force_reuse','fused_optimizer','loss_backend','fused_swiglu','allocator'))
 matched={'physical_batch':50,'optimization':{k:a[k] for k in ('chunk_tokens','force_reuse','fused_optimizer','loss_backend','fused_swiglu','allocator')},'allocated_reduction_fraction':1-b['capacity']['peak_allocated_bytes']/a['capacity']['peak_allocated_bytes'],'reserved_reduction_fraction':1-b['capacity']['peak_reserved_bytes']/a['capacity']['peak_reserved_bytes'],'midpoint_tps_ratio':b['timing']['targets_per_second']/a['timing']['targets_per_second'],'single_probe_each':True}
 result={'archive':str(ARCHIVE),'archive_sha256':sha(ARCHIVE.read_bytes()),'zip_crc_ok':True,'members':members,'all_standalone_results_equal_suite':True,'embedded_receipt_matches_delivered_v2':True,'source_config_data_runtime_checks_pass':True,'all_25_frozen_sources_unchanged':True,'receipt':receipt,'totals':totals,'gates':gate_summary,'maxima':max_summary,'batch50_decisions':suite['decisions'],'capacity_ratio_selected_paths':814/285,'matched_batch50_ce_only':matched,'selected_probes':condensed,'validation_loss_ceiling':5.56229485,'full_training_completed':False,'loss_equivalence_or_savings_proven':False}
 OUT.mkdir(exist_ok=True)
 (OUT/'capacity-results-audit-2026-10-10.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('totals','maxima','matched_batch50_ce_only','capacity_ratio_selected_paths')},indent=2))
