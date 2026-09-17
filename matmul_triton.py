import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,  # pointers to matrices
        M, N, K,    # dimensions
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    """
    C = A x B ((M x K) @ (K x N) => (M, N))
    """
    # Convert linear program ID pid into 2D coordinates (pid_m, pid_n), visit output tiles in a cache friendly manner
    pid = tl.program_id(axis=0)
    # determine number of program instances along M & N axes
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # compute number of programs in a group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Find the group id of the program we're in
    group_id = pid // num_pid_in_group
    # determine the row id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # if number of programs in M isn't divisible by GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # Determine pid_m, pid_n
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Copied from triton tutorial https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Multi-dimensional pointer arithmetic
    offs_am = ((pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = ((pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # a_ptrs shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    # b_ptrs shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Initialize accumulator with zeros of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A, B blocks and mask by checking the K dimension
        # to make sure that loaded elements within the range (avoid OOB error)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # After each iteration move both A&B tile to next slice along K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        # compute the dot of a&b blocks
        # After all K tiles accumulator contains the output block of C
        acc += tl.dot(a, b)
    
    #Provide arbitrary activation function while the acc still in FP32
    if ACTIVATION == "relu":
        acc = relu(acc)
    # cast into fp16 dtype
    c = acc.to(dtype=tl.float16)

    # Write back the output block computed by program instance
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, c_mask)

@triton.jit
def relu(x):
    return tl.where(x > 0, x, 0)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

# Unit Test
torch.manual_seed(0)
a = torch.rand((256, 256), dtype=torch.float16) - 0.5
b = torch.rand((256, 256), dtype=torch.float16) - 0.5
torch_res = torch.matmul(a, b)
triton_res = matmul(a, b)
print(f"torch output with fp16 inputs: {torch_res}")
print(f"triton output with fp16 inputs: {triton_res}")
print(f"Abs difference between torch and triton outputs: {torch.allclose(torch_res, triton_res)}")

if torch.allclose(torch_res, triton_res, atol=1e-3, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    a = torch.rand((256, 256), dtype=torch.float16) - 0.5
    b = torch.rand((256, 256), dtype=torch.float16) - 0.5
    a = a.to(torch.float8_e5m2)
    # transpose b for efficiency
    b = b.T
    b = b.to(torch.float8_e5m2)
    torch_res = torch.matmul(a.to(dtype=torch.float16), b.to(dtype=torch.float16))
    triton_res = matmul(a, b)
    print(f"torch output with fp16 inputs: {torch_res}")
    print(f"triton output with fp16 inputs: {triton_res}")

    if torch.allclose(torch_res, triton_res, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("Triton and Torch differ")


