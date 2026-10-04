import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_reversible_study as study


def test_environment_reports_all_build_differences():
    expected={'torch':'2.11.0+cu128','cuda_runtime':'12.8','numpy':'2.1.3','python':'3.13.15'}
    actual={**expected,'torch':'2.11.0+cu130','cuda_runtime':'13.0'}
    failures=study.environment_mismatches(expected,actual)
    assert failures==['torch: 2.11.0+cu130 != 2.11.0+cu128','cuda_runtime: 13.0 != 12.8']
    assert study.environment_mismatches(expected,expected)==[]


def unused_preflight(root):
    root.mkdir()
    (root/'source-manifest.json').write_text(json.dumps({'old-source':'old-hash'}))
    (root/'configs').mkdir()
    (root/'configs/midpoint_matched.json').write_text('old frozen config')
    (root/'matched-orchestration.log').write_text('environment mismatch')


def test_unused_preflight_is_preserved_before_refresh(tmp_path):
    root=tmp_path/'study';unused_preflight(root)
    study.refresh_unused_preflight(root,{'new-source':'new-hash'})
    history=next((root/'preflight-history').iterdir())
    assert (history/'source-manifest.json').is_file()
    assert (history/'configs/midpoint_matched.json').read_text()=='old frozen config'
    assert (root/'matched-orchestration.log').read_text()=='environment mismatch'
    assert not (root/'source-manifest.json').exists()


@pytest.mark.parametrize('evidence',['metrics.jsonl','latest.pt'])
def test_started_training_prevents_source_refresh(tmp_path,evidence):
    root=tmp_path/'study';unused_preflight(root)
    run=root/'runs/midpoint';run.mkdir(parents=True)
    (run/evidence).write_text('training evidence')
    with pytest.raises(RuntimeError,match='source changed after training started'):
        study.refresh_unused_preflight(root,{'new-source':'new-hash'})
    assert (root/'source-manifest.json').is_file()
    assert (root/'configs/midpoint_matched.json').is_file()


def test_environment_failure_does_not_freeze_study(tmp_path,monkeypatch):
    config=tmp_path/'configs/baseline_colab.json';config.parent.mkdir();config.write_text('{}')
    monkeypatch.setattr(sys,'argv',['study','matched','--artifacts',str(tmp_path)])
    monkeypatch.setattr(study,'completed',lambda *args,**kwargs:{})
    monkeypatch.setattr(study,'startup',lambda run:{'config_sha256':study.sha(config),'environment':{}})
    def mismatch(expected):
        raise RuntimeError('environment mismatch')
    monkeypatch.setattr(study,'check_environment',mismatch)
    with pytest.raises(RuntimeError,match='environment mismatch'):
        study.main()
    assert not (tmp_path/'reversible-v1').exists()
