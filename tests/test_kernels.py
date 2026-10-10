"""Kernel runtime checks; CUDA skips are not evidence of GPU correctness."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_kernels_reject_cpu_instead_of_silent_fallback():
    pytest.importorskip('triton')
    from reversibility.kernels import cross_entropy_sum, FusedSwiGLU
    with pytest.raises(ValueError, match='CUDA'):
        cross_entropy_sum(torch.zeros(3, 31), torch.tensor([0, 1, -100]))
    with pytest.raises(ValueError, match='CUDA'):
        FusedSwiGLU.apply(torch.zeros(3, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_tiled_ce_full_vocabulary_tail_mask_and_scaled_backward(dtype):
    from reversibility.kernels import cross_entropy_sum
    torch.manual_seed(41)
    a = torch.randn(17, 50257, device='cuda', dtype=dtype, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    y = torch.tensor([0, 1023, 1024, 50256, -100] + list(range(12)), device='cuda')
    with torch.autocast('cuda', enabled=dtype == torch.float16, dtype=torch.float16):
        expected = F.cross_entropy(a, y, reduction='sum', ignore_index=-100)
    actual = cross_entropy_sum(b, y)
    (expected * .03125).backward(); (actual * .03125).backward()
    torch.testing.assert_close(actual, expected.float(), rtol=.002, atol=.002)
    relative = (a.grad - b.grad).norm() / a.grad.norm()
    assert relative < (.02 if dtype == torch.float16 else 2e-5)
    assert torch.count_nonzero(b.grad[4]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_swiglu_tail_and_backward(dtype):
    from reversibility.kernels import FusedSwiGLU
    torch.manual_seed(42)
    a = torch.randn(3, 7, 66, device='cuda', dtype=dtype, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    gate, value = a.chunk(2, dim=-1)
    expected = F.silu(gate) * value
    actual = FusedSwiGLU.apply(b)
    upstream = torch.randn_like(actual)
    expected.backward(upstream); actual.backward(upstream)
    tolerance = .003 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(b.grad, a.grad, rtol=tolerance, atol=tolerance)
