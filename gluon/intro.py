import torch
import triton
import pytest
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    out = gl.load(in_ptr)
    gl.store(out_ptr, out)

def copy_scalar(input, output):
    grid = (1,)
    copy_scalar_kernel[grid](input, output)

def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_allclose(input, output, atol=0.0, rtol=0.0)

@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, BLOCK_SIZE, NUM_ELE: tl.constexpr):
    # Identify the program id we're running on
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, NUM_ELE)
    for i in range(start, end):
        out = gl.load(in_ptr + i)
        gl.store(out_ptr + i, out)

def memcpy(input, output, BLOCK_SIZE):
    NUM_ELE = input.numel()
    grid = (triton.cdiv(NUM_ELE, BLOCK_SIZE))
    memcpy_kernel(input, output, BLOCK_SIZE, NUM_ELE, num_warps=1)

@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
def test_memcpy(BLOCK_SIZE, NUM_ELE):
    input = torch.randn(NUM_ELE, device="cuda")
    output = torch.empty_like(input)
    memcpy(input, output, BLOCK_SIZE)
    torch.testing.assert_allclose(input, output, atol=0.0, rtol=0.0)
