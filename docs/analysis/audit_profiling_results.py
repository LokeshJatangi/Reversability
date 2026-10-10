from pathlib import Path
import argparse,collections,hashlib,json,pickle,zipfile

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise ValueError(f'Unexpected pickle global: {module}.{name}')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())

def classify_kernel(op,tokens,vocab,width):
    name=op['name'];dims=op.get('args',{}).get('Input Dims',[])
    if name in ['aten::mm','aten::addmm','aten::bmm'] and any(isinstance(x,list) and vocab in x for x in dims):
        if dims[:2]==[[tokens,width],[width,vocab]]:return 'head_forward_gemm'
        if dims[:2]==[[vocab,tokens],[tokens,width]]:return 'head_weight_gradient_gemm'
        if dims[:2]==[[tokens,vocab],[vocab,width]]:return 'head_hidden_gradient_gemm'
        return 'other_vocabulary_matrix_op'
    if any(isinstance(x,list) and x==[tokens,vocab] for x in dims):return 'vocabulary_loss_casts_gradient_buffers'
    if 'attention' in name:return 'attention_kernels'
    return 'other_transformer_optimizer'

def snapshot_evidence(path,reported_peak):
    with path.open('rb') as handle:s=RestrictedUnpickler(handle).load()
    events=s['device_traces'][0]
    assert len(events)<100000,'Allocation history may have truncated'
    active={block['address']:{'size':block['size'],'frames':block['frames'],'preexisting':True}
        for segment in s['segments'] for block in segment['blocks'] if block['state']=='active_allocated'}
    final_addresses=set(active)
    for event in reversed(events):
        if event['action']=='alloc':
            assert event['addr'] in active;del active[event['addr']]
        elif event['action']=='free_requested':
            assert event['addr'] not in active
            active[event['addr']]={'size':event['size'],'frames':[],'preexisting':True}
    total=sum(e['size'] for e in active.values());peak=total;at_peak=dict(active);peak_index=-1
    for i,event in enumerate(events):
        if event['action']=='alloc':
            assert event['addr'] not in active
            active[event['addr']]=event;total+=event['size']
        elif event['action']=='free_requested':total-=active.pop(event['addr'])['size']
        if total>peak:peak=total;at_peak=dict(active);peak_index=i
    assert set(active)==final_addresses
    assert abs(peak-reported_peak)/reported_peak<.01,'Requested-byte reconstruction does not match measured peak closely'
    large=[]
    for event in at_peak.values():
        if event['size']<300000000:continue
        frames=event.get('frames',[]);names=' '.join(f['name'] for f in frames)
        if 'structured_nll_loss_backward' in names:role='dense NLL gradient (FP32-shaped bytes)'
        elif 'at::autocast::WrapFunction_' in names and 'nll_loss' in names:role='autocast NLL input cast (FP32-shaped bytes)'
        elif 'cross_entropy_loss_symint' in names:role='log-softmax result (FP16-shaped bytes)'
        else:role='output projection logits (FP16-shaped bytes)'
        large.append({'bytes':event['size'],'role':role,'allocation_frames':frames})
    assert len(large)==4
    large_sum=sum(e['bytes'] for e in large)
    return {'trace_event_count':len(events),'near_peak_event_index':peak_index,
        'requested_byte_replay_peak':peak,'reported_allocated_peak':reported_peak,
        'replay_difference_bytes':peak-reported_peak,
        'replay_note':'Trace request sizes and final allocator block sizes differ by rounding/reuse. Near-peak replay is within 1%; use measured CUDA stats as peak authority.',
        'four_large_simultaneous_buffers':large,'four_buffers_bytes':large_sum,
        'four_buffers_fraction_of_reported_peak':large_sum/reported_peak}

def audit(folder,archive,repo):
    receipt=load(next(folder.glob('*receipts.json')))
    root=next(folder.glob('*/suite.json')).parent;suite=load(root/'suite.json')
    assert suite['status']=='complete' and not suite['cpu_smoke_only'] and len(suite['records'])==9
    assert receipt['declared_environment_differences']=={}
    assert receipt['source_commit']=='d071e1e2d8e4575d2fc9ca5ccb1ef4e3d5e185b7'
    assert receipt['profile_sha256']==digest(repo/'scripts/profile_training.py')
    frozen=load(repo/'results/recorded/reversible-v1/source-manifest.json')
    assert all(digest(repo/p)==h for p,h in frozen.items())
    zip_files=0
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in z.infolist():
            if entry.is_dir():continue
            path=folder/entry.filename
            assert path.is_file() and path.read_bytes()==z.read(entry)
            zip_files+=1
    results={}
    for name,method,batch in [('baseline_29','baseline',29),('midpoint_29','midpoint',29),('midpoint_33','midpoint',33)]:
        records={mode:load(root/name/mode/'result.json') for mode in ['timing','trace','memory']}
        for mode,result in records.items():
            assert result['status']=='complete' and not result['cpu_smoke_only']
            assert result['parameters']==20340736
            assert result['config']['physical_batch_size']==batch
            assert result['config']['method']==method
            assert result['config']['sequence_length']==512
            assert result['config']['gradient_accumulation_steps']==2
            assert result['manifest_sha256']==receipt['data']['manifest_sha256']
            assert result['config_file_sha256']==digest(repo/'results/recorded/configs/baseline_colab.json')
            for source,sha in result['source_sha256'].items():assert digest(repo/source)==sha
            expected_runtime=receipt['runtime']
            env=result['environment']
            assert env['torch']==expected_runtime['torch'] and env['numpy']==expected_runtime['numpy']
            assert env['cuda_runtime']==expected_runtime['cuda_runtime']
            assert env['accelerator']['name']==expected_runtime['gpu']
        trace=load(root/name/'trace/trace.json')['traceEvents']
        ops={e.get('args',{}).get('External id'):e for e in trace if e.get('cat')=='cpu_op'}
        phases=[e for e in trace if e.get('cat')=='user_annotation' and e.get('name','').startswith('profile/reversible/')]
        buckets=collections.Counter();phase_times=collections.Counter();kernel_count=0
        for event in trace:
            if event.get('cat')!='kernel':continue
            kernel_count+=1
            op=ops[event['args']['External id']]
            buckets[classify_kernel(op,batch*512,50257,256)]+=event['dur']
            parents=[r for r in phases if r['pid']==op['pid'] and r['tid']==op['tid']
                and r['ts']<=op['ts'] and op['ts']+op['dur']<=r['ts']+r['dur']+1]
            if parents:phase_times[min(parents,key=lambda r:r['dur'])['name']]+=event['dur']
        head_loss=sum(value for key,value in buckets.items() if key.startswith('head_') or key=='vocabulary_loss_casts_gradient_buffers')
        dtypes=[]
        for event in trace:
            if event.get('cat')=='cpu_op' and event.get('name') in ['aten::_log_softmax','aten::_log_softmax_backward_data','aten::cross_entropy_loss','aten::nll_loss_forward','aten::nll_loss_backward']:
                args=event.get('args',{})
                if any(isinstance(d,list) and 50257 in d for d in args.get('Input Dims',[])):
                    row={'operator':event['name'],'input_types':args.get('Input type'),'input_shapes':args.get('Input Dims')}
                    if row not in dtypes:dtypes.append(row)
        rows=records['memory']['memory_regions'];grouped={}
        for row in rows:
            group=grouped.setdefault(row['region'],{'calls':0,'maximum_peak_allocated_bytes':0,'maximum_rise_bytes':0})
            group['calls']+=1;group['maximum_peak_allocated_bytes']=max(group['maximum_peak_allocated_bytes'],row['peak_allocated_bytes'])
            group['maximum_rise_bytes']=max(group['maximum_rise_bytes'],row['peak_allocated_bytes']-row['start_allocated_bytes'])
        timing=records['timing']['timing']
        assert timing['measured_valid_targets']==20*batch*512*2 and len(timing['update_seconds'])==20
        kernel_sum=sum(buckets.values())
        results[name]={'timing':timing,'kernel_count':kernel_count,'kernel_microseconds_by_exclusive_bucket':dict(buckets),
            'kernel_total_microseconds':kernel_sum,'head_loss_kernel_microseconds':head_loss,
            'head_loss_fraction_of_kernel_time':head_loss/kernel_sum,'reversible_phase_kernel_microseconds':dict(phase_times),
            'observed_operator_dtypes':dtypes,'memory_regions_grouped':grouped,
            'allocator_evidence':snapshot_evidence(root/name/'memory/allocator_snapshot.pickle',records['memory']['memory_diagnostic_peak']['peak_allocated_bytes']),
            'loss_max_spread_across_fresh_processes':max(r['losses'][0] for r in records.values())-min(r['losses'][0] for r in records.values()),
            'file_sha256':{mode:digest(root/name/mode/'result.json') for mode in records},
            'trace_sha256':digest(root/name/'trace/trace.json'),'snapshot_sha256':digest(root/name/'memory/allocator_snapshot.pickle')}
    b=results['baseline_29']['timing']['targets_per_second'];m=results['midpoint_29']['timing']['targets_per_second'];x=results['midpoint_33']['timing']['targets_per_second']
    return {'archive_sha256':digest(archive),'archive_files_verified':zip_files,'receipt':receipt,'frozen_source_hashes_verified':len(frozen),'passes_verified':9,'results':results,
      'comparisons':{'matched_runtime_penalty_fraction':b/m-1,'maximum_runtime_penalty_fraction':b/x-1,'batch33_throughput_change_fraction':x/m-1,
        'matched_allocated_memory_saving_fraction':1-results['midpoint_29']['timing']['peak_allocated_bytes']/results['baseline_29']['timing']['peak_allocated_bytes']},
      'methodology':'Count each raw CUDA kernel once, join to the CPU leaf op by External id, classify head matrix/casts by tensor shape. Sum overlapping annotation regions only after assigning each kernel to the smallest containing reversible CPU range. Allocator pickle accepts only primitive structures; replay alloc/free_requested to identify simultaneously live large buffers. Original measured allocated stats are authoritative.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.folder,args.archive,args.repo)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified',result['passes_verified'],'passes;',result['archive_files_verified'],'ZIP files;',result['frozen_source_hashes_verified'],'frozen source hashes')
    for name,row in result['results'].items():
        print(name,'head/loss kernel share',round(100*row['head_loss_fraction_of_kernel_time'],2),'large-buffer/peak share',round(100*row['allocator_evidence']['four_buffers_fraction_of_reported_peak'],2))
