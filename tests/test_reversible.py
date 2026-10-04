import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reversibility import BaselineLM, ModelConfig, ReversibleLM
from validate_reversible import check_case
from test_baseline import _write_tiny_training_fixture, _run_training
import json

@pytest.mark.parametrize('method',['midpoint','euler'])
@pytest.mark.parametrize('precision',['fp64','fp32'])
def test_reconstruction_gradients_and_update(method,precision):
    check_case(method,precision,'cpu',3,17)

@pytest.mark.parametrize('method',['midpoint','euler'])
def test_initialization_mapping_and_parameter_count(method):
    config=ModelConfig(vocab_size=101,n_layers=3,d_model=32,n_heads=2,d_ff=64)
    torch.manual_seed(1337);base=BaselineLM(config)
    torch.manual_seed(1337);rev=ReversibleLM(config,method)
    assert base.parameter_count()==rev.parameter_count()
    for name,p in base.state_dict().items(): torch.testing.assert_close(p,rev.state_dict()[name],rtol=0,atol=0)

@pytest.mark.parametrize('method',['midpoint','euler'])
def test_inverse_stack_retains_only_final_pair_and_parameters(method):
    model=ReversibleLM(ModelConfig(vocab_size=31,n_layers=7,n_heads=1,d_model=8,d_ff=16),method)
    x=torch.randn(2,4,8,requires_grad=True)
    y=model.core(x)
    # Core output is one of two Function outputs: two activation tensors + parameter refs.
    saved=y.grad_fn.saved_tensors
    assert len(saved)==2+len(tuple(model.blocks.parameters()))
    assert [tuple(t.shape) for t in saved[:2]]==[(2,4,8),(2,4,8)]

@pytest.mark.parametrize('method',['midpoint','euler'])
def test_reversible_resume(tmp_path,monkeypatch,method):
    paths=[_write_tiny_training_fixture(tmp_path,n) for n in ['uninterrupted','resumed']]
    for p in paths:
        cfg=json.loads(p.read_text());cfg.update(method=method,backward_mode='reconstruct',step_size=.5)
        p.write_text(json.dumps(cfg))
    _run_training(monkeypatch,['--config',paths[0],'--device','cpu'])
    _run_training(monkeypatch,['--config',paths[1],'--device','cpu','--stop-after-targets',16])
    checkpoint=tmp_path/'resumed/latest.pt'
    _run_training(monkeypatch,['--config',paths[1],'--device','cpu','--resume',checkpoint])
    a=torch.load(tmp_path/'uninterrupted/latest.pt',weights_only=False)
    b=torch.load(checkpoint,weights_only=False)
    assert a['committed_targets']==b['committed_targets']==32
    for name,t in a['model'].items():torch.testing.assert_close(t,b['model'][name],rtol=0,atol=0)
    for pid,state in a['optimizer']['state'].items():
        for name,t in state.items():
            if torch.is_tensor(t):torch.testing.assert_close(t,b['optimizer']['state'][pid][name],rtol=0,atol=0)
            else:assert t==b['optimizer']['state'][pid][name]
