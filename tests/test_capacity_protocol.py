import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_batch_capacity as capacity

def test_capacity_stops_on_numerical_failure_and_preserves_attempts(tmp_path,monkeypatch):
    config=tmp_path/'config.json';config.write_text('{}')
    report=tmp_path/'capacity.json'
    monkeypatch.setattr(sys,'argv',['capacity','--config',str(config),'--output',str(report)])
    monkeypatch.setattr(capacity,'run_candidate',lambda args,batch:{'physical_batch_size':batch,'status':'numerical_failure'})
    with pytest.raises(RuntimeError,match='non-memory reason'):capacity.main()
    attempts=json.loads(report.with_suffix('.attempts.json').read_text())
    assert len(attempts['attempts'])==1
    assert not report.exists()


def test_capacity_selects_largest_verified_and_records_accumulation(tmp_path,monkeypatch):
    cfg={'sequence_length':512,'precision':'fp16','model':{},'gradient_accumulation_steps':2}
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg));report=tmp_path/'capacity.json'
    monkeypatch.setattr(sys,'argv',['capacity','--config',str(config),'--output',str(report),'--max-batch','64'])
    def fake(args,batch):
        return {'physical_batch_size':batch,'status':'pass' if batch<=29 else 'insufficient_headroom',
                'gpu_name':'test','gradient_accumulation_steps':2}
    monkeypatch.setattr(capacity,'run_candidate',fake)
    capacity.main();out=json.loads(report.read_text())
    assert out['selected_physical_batch_size']==29
    assert not out['search_was_capped']
    assert any(x['physical_batch_size']==30 for x in out['attempts'])
