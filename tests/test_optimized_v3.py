import copy
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reversibility import build_model
from reversibility.optimized import chunked_backward
from reversibility.optimized_v3 import install_v3,cached_head_backward,TiledMLP,SplitMidpoint
from reversibility.reversible import forward_pair


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
@pytest.mark.parametrize('core',['reference','reuse','split'])
@pytest.mark.parametrize('tile',[0,5])
def test_full_batch_v3_tied_gradients_masks_and_two_updates(dtype,core,tile):
    config={'model':{'vocab_size':31,'max_sequence_length':8,'n_layers':3,'n_heads':2,'d_model':8,'d_ff':16},'method':'midpoint'}
    torch.manual_seed(71);reference=build_model(config).to(dtype);candidate=copy.deepcopy(reference)
    install_v3(candidate,core_mode=core,mlp_tokens=tile)
    a=torch.optim.AdamW(reference.parameters(),lr=.0006);b=torch.optim.AdamW(candidate.parameters(),lr=.0006)
    for step in range(2):
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        ids=torch.randint(31,(5,8));labels=ids.roll(-1,1);labels[-1,:]=-100;labels[0,-1]=-100
        denominator=int((labels!=-100).sum())
        expected=F.cross_entropy(reference(ids).flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum')
        (expected/denominator).backward()
        actual=cached_head_backward(candidate,ids,labels,denominator,7)
        torch.testing.assert_close(actual,expected.detach().float(),rtol=2e-6,atol=2e-5)
        tolerance=1e-10 if dtype==torch.float64 else 5e-5
        for p,q in zip(reference.parameters(),candidate.parameters()):
            assert torch.isfinite(q.grad).all()
            torch.testing.assert_close(p.grad,q.grad,rtol=tolerance,atol=tolerance*.01)
        a.step();b.step()
    assert candidate.lm_head.weight is candidate.token_embedding.weight


@pytest.mark.parametrize('h',[.25,.5,.75])
def test_split_state_and_vjp_against_unrolled(h):
    torch.manual_seed(73)
    a=build_model({'model':{'vocab_size':31,'n_layers':3,'n_heads':2,'d_model':8,'d_ff':16},'method':'midpoint'}).double();b=copy.deepcopy(a)
    p=torch.randn(3,4,8,dtype=torch.float64,requires_grad=True);q=torch.randn_like(p,requires_grad=True)
    rp=p.detach().clone().requires_grad_(True);rq=q.detach().clone().requires_grad_(True)
    x,y=p,q
    for block in a.blocks:x,y=forward_pair(block,x,y,'midpoint',h)
    u,v=SplitMidpoint.apply(rp,rq,b.blocks,h,*b.blocks.parameters())
    gp=torch.randn_like(p);gq=torch.randn_like(q)
    torch.autograd.backward((x,y),(gp,gq));torch.autograd.backward((u,v),(gp,gq))
    torch.testing.assert_close(x,u,rtol=0,atol=0);torch.testing.assert_close(y,v,rtol=0,atol=0)
    for left,right in [(p.grad,rp.grad),(q.grad,rq.grad),*[(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters()) if x.grad is not None]]:
        torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-11)


def test_tiled_mlp_workspace_and_entire_input(tmp_path,monkeypatch):
    import reversibility.optimized_v3 as v3
    sizes=[];original=v3.mlp_value
    def observe(x,*args):sizes.append(x.shape[0]);return original(x,*args)
    monkeypatch.setattr(v3,'mlp_value',observe)
    x=torch.randn(50,8,8,requires_grad=True);up=torch.randn(32,8,requires_grad=True);down=torch.randn(8,16,requires_grad=True)
    y=TiledMLP.apply(x,up,down,37,False);y.square().mean().backward()
    assert y.shape==(50,8,8)
    assert len(sizes)==2*((400+36)//37) and max(sizes)<=37
    assert x.grad.shape==x.shape


def test_v3_norm_rejects_cpu():
    pytest.importorskip('triton')
    from reversibility.kernels_v3 import FusedRMSNorm
    with pytest.raises(ValueError,match='CUDA FP32'):
        FusedRMSNorm.apply(torch.ones(3,8),torch.ones(8),1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA runtime required')
def test_rmsnorm_full_tail_forward_backward():
    from reversibility.kernels_v3 import FusedRMSNorm
    torch.manual_seed(74)
    x=torch.randn(3,23,256,device='cuda',requires_grad=True);w=torch.randn(256,device='cuda',requires_grad=True)
    u=x.detach().clone().requires_grad_(True);v=w.detach().clone().requires_grad_(True)
    expected=F.rms_norm(x,(256,),w,eps=torch.finfo(x.dtype).eps)
    actual=FusedRMSNorm.apply(u,v,torch.finfo(x.dtype).eps)
    g=torch.randn_like(actual);expected.backward(g);actual.backward(g)
    torch.testing.assert_close(expected,actual,rtol=5e-5,atol=5e-6)
    torch.testing.assert_close(x.grad,u.grad,rtol=5e-5,atol=5e-6)
    torch.testing.assert_close(w.grad,v.grad,rtol=5e-5,atol=5e-5)


@pytest.mark.parametrize('headroom',[.10,.05])
def test_default_v3_reviews_probes_without_maximum_search(tmp_path,monkeypatch,headroom):
    from argparse import Namespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import benchmark_midpoint_v3 as bench
    calls=[]
    def run(command,**kwargs):
        def arg(key):return command[command.index(key)+1]
        output=Path(arg('--output'));output.mkdir()
        batch=int(arg('--batch'));method=arg('--method')
        memory=batch*1000000 + (0 if '--rmsnorm' in command else 10000000)
        result={'status':'complete','gate':{'passed':True,'passing_chunks':[1024]},
            'config':{'physical_batch_size':batch,'method':method},
            'capacity':{'passed':True,'peak_reserved_bytes':memory,'peak_allocated_bytes':memory-1000},
            'timing':{'targets_per_second':100},'v3_features':{},'headroom':float(arg('--headroom'))}
        (output/'result.json').write_text(__import__('json').dumps(result));calls.append(command)
        return Namespace(returncode=0)
    monkeypatch.setattr(bench.subprocess,'run',run)
    bench.suite(Namespace(output=tmp_path/'run',config=tmp_path/'config',data_dir=tmp_path/'data',
        cpu_smoke=False,headroom=headroom,variant_set='all',capacity_search=False,warmup=5,updates=20))
    data=__import__('json').loads((tmp_path/'run/suite.json').read_text())
    assert data['searches']=={} and not data['capacity_search_requested']
    selected=data['selected_midpoint']
    assert selected['verified_probe_batch']==(814 if headroom==.10 else 1024)
    assert not selected['absolute_maximum_established']
    confirms=[c for c in calls if '/confirm-' in c[c.index('--output')+1]]
    assert len(confirms)==3 and '--checkpoint' in confirms[2]
    assert all(c[c.index('--headroom')+1]==str(headroom) for c in calls)
