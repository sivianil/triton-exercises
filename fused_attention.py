import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    desc_k, desc_v,
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr
                    ):
    # range of pointers handled by each stage
    if STAGE == 1:
        low, high = 0, start_m * BLOCK_M
    elif STAGE == 2:
        low, high = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        low = tl.multiple_of(low, BLOCK_M)
    else:
        low, high = 0, N_CTX
    # offsets for key/value 
    offsetk_y = offset_y + low
    offsetv_y = offset_y + low

    # Iterate through the key & value blocks and update accumulator
    for start_n in range(low, high, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # load key block
        k = desc_k.load([offsetk_y, 0]).T       # (HEAD_DIM, BLOCK_N)
        # compute qk
        qk = tl.dot(q, k)
        if STAGE == 2:  # casual masking
            mask = offs_m[:, None] >= start_n + offs_n[None, :]      # start_n is token offset for the key block; tokens comes next set to False or 0
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            # compute the local maximum
            m_ij = tl.maximum(m_i, tl.max(qk,1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk,1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        
        # compute the exponentials of the scores
        p = tl.math.exp2(qk)

        # correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc)  # computes the dot product (p, v) & add it to acc

        # update m_i, l_i
        l_i = alpha * l_i + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(sm_scale, M,
              Z, H,  # Z refers to batch_size and H to no of heads
              q, k, v, o, N_CTX,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              ):
    dtype = tl.float16
    # Identify the query token block and batch/head instance we're on
    start_m = tl.program_id(0)    # which query block index we're on
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Reduce the 4D tensor into 2D tensor
    y_dim = Z * H * N_CTX

    desc_q = tl.make_tensor_descriptor(q,
                                        shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(k,
                                        shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(v,
                                        shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(o,
                                        shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1],
                                        block_shape=[BLOCK_M, HEAD_DIM])
    
    # Offsets of y_dim
    """
    Let assume M has shape (16, 64, 1024) stored contiguously in memory
    For one batch, there are H x N_CTX elements => 64 * 1024 => 65536 elements
    To move from Z = 0 to 1, we need to jump H * N_CTX elements
    Within batch, each head owns N_CTX elements
    offset_y gives element offset for (Z, H) pair
    off_z = 2, off_h = 32
    offset_y = 2 * (64 * 1024) + 32 * 1024 => 163840 
    so M[2, 32, 0] located at offset 163840
    """
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    
    """
    start_m * BLOCK_M gives query token offset
    BLOCK_M = 128 for N_CTX = 1024, total Q block instances => N_CTX // BLOCK_M 
    start_m = 0 -> token 0
    start_m = 1 -> token 128
    ........................
    start_m = 7 -> token 896
    qo_offset_y points to the first token of the current Q block for that batch/head
    """
    qo_offset_y = offset_y + start_m * BLOCK_M     
    
    # Initialize offsets for the attention matrix (masking/index calculations)
    # for casual attention, offsets_m refers to query positions and offsets_n to key positions
    offs_m = start_m * BLOCK_M + tl.range(0, BLOCK_M)
    offs_n = tl.range(0, BLOCK_N)

    # Initialize pointers to softmax statistics m, l, acc
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    # Triton Kernel computes exponentials in base2(exp2) instead of e (exp)
    # triton computes 2**x becoz tl.exp2() used inside the attn kernel
    """
    Normal attention:
        QK × (1/√d)
               ↓
             exp(x)

    This kernel:
        QK × (1/√d) × 1/ln(2)
               ↓
             exp2(x)
    """
    qk_scale *= 1.44269504 

    # Load query block from HBM onto SRAM
    q = desc_q.load([qo_offset_y, 0])    # loads [BLOCK_M, HEAD_DIM] tile

    # stage 1: off-band
    # For causal = TRUE, STAGE = 3 
    # For causal = FALSE, STAGE = 1
    """
    STAGE = 3 executes both stages; _attn_fwd_inner gets STAGE = 1 & 2
    why split casual attn into two stages?
    For each Q block, K blocks divides into off-band and on-band
    on-band (causal boundary) q_idx >= k_idx
    """
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, N_CTX
                                        )
    if STAGE & 2:   # bit masking 
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                        desc_k, desc_v,
                                        offset_y, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        2, offs_m, offs_n, N_CTX
                                        )
    # compute LSM statistics for each Q block
    m_i += tl.math.log(l_i)
    acc /= l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))

def _attn_backward_preprocess(
        O, DO, 
        Delta, 
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    offs_zh = tl.program_id(1) # which batch/head pair we're on
    # offsets of the query token block we're running on 
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + offs_zh * (N_CTX * HEAD_DIM) + offs_m[:, None] * HEAD_DIM + offs_n[:, None])
    do = tl.load(DO + offs_zh * (N_CTX * HEAD_DIM) + offs_m[:, None] * HEAD_DIM + offs_n[:, None]).to(tl.float32)
    delta = tl.sum(o*do, axis=1)
    tl.store(Delta + offs_zh * N_CTX + offs_m, delta)
    
def _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, SM_SCALE,
        DO,
        M, DELTA,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M1: tl.constexpr,
        BLOCK_N1: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        start_n, start_m, num_steps,
        MASK=tl.constexpr,
):
    """
    Step1: Find the token offsets to load pointers of Q and DO
    Step2: Iterate over the N_CTX//BLOCK_M1 steps 
        a: For each step we read Q/DO from HBM 
        b: Load M from HBM
        c: compute qkT
        d: compute pT = tl.exp2(qkT - m)
        e: Load DO and compute dv i.e.,(dv += pT*DO)
        f: load Di
        g: compute dpT (dpT = DOT*v)
        h: compute dsT => pT*(dpT - Di)                                        
        i: compute dk (dk += dsT * qT)
        j: Increment pointers curr_m, qT_ptrs, do_ptrs
    Step3: Return dk, dv
    """
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N1)  # only required for masking

    qT_ptrs = Q + offs_m[:, None] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1  # step size of the block
    for block_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)

        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)

        do = tl.load(do_ptrs)
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)

        D_i = tl.load(DELTA + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - D_i[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(Q))

        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    
    return dk, dv

def _attn_bwd_dq(
        dq, q, K, V,
        do, m, DELTA,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M2: tl.constexpr,
        BLOCK_N2: tl.constexpr, 
        HEAD_DIM: tl.constexpr,
        start_m, start_n, num_steps,
        MASK: tl.constexpr,
    ):
    """
    Iterate over the (N_CTX//BLOCK_N2) passes of DQ
    For each pass we read/write all DQ from/to HBM
    For a fixed K/V block, each contributes something to DQ
    Mental Model:
    Pass0:  Load K0, V0 
            read DQ from HBM onto SRAM
            add the dot product of DS and K0 to DQ
            write DQ to HBM
    
    Pass1:  Load K1, V1 
            read DQ from HBM onto SRAM
            add the dot product of DS and K0 to DQ
            write DQ to HBM
    ---------------------------------
    PassN:  Load KN, VN 
            read DQ from HBM onto SRAM
            add the dot product of DS and K0 to DQ
            write DQ to HBM
    """
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    kT_ptrs = K + offs_n[:, None] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[:, None] * stride_tok + offs_k[:, None] * stride_d

    # Load D_i
    D_i = tl.load(DELTA + offs_m)
    tl.static_assert(BLOCK_M2%BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for block_idx in range(num_steps):
        # load kT, vT
        kT = tl.load(kT_ptrs)
        vT = tl.laod(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m[None, :])
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(do, tl.trans(vT)).to(tl.float32)
        ds = p * (dp - D_i[None, :])
        ds = ds.to(tl.float16)
        dq += tl.dot(ds, tl.trans(kT))

        # Increment pointers
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq

def _attn_backward(
        Q, K, SM_SCALE, V,
        DO,
        DQ, DK, DV,
        M, DELTA,
        stride_z, stride_h, stride_tok, stride_d,  # strides across batch, head, context and head dim
        H, N_CTX,
        BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
        BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
        BLK_SLICE_FACTOR: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr, 
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    pid_0 = tl.program_id(0)  # identify the program instance we're running on
    bhid = tl.program_id(1) # idenitfy the batch/head 
    # token offset for a given batch/head pair
    off_zh = (bhid * N_CTX).to(tl.int64)
    adj = (stride_z * (bhid // H) + stride_h * (bhid % H)).to(tl.int64)

    # offset pointers for batch/head pair
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_zh
    D += off_zh

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid_0 * BLOCK_N1
    start_m = 0

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # Initialize dk/dv tensors with shape (BLOCK_N1, HEAD_DIM)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype= tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype= tl.float32)

    # Load K and V from HBM into on-chip SRAM: they stay throughout the inner loop
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    # compute dK/dV for non-masked blocks
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, SM_SCALE,
        DO,
        M, DELTA,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        MASK=False,
    )

    # Write back DK, DV
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk *= SM_SCALE
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)


    # Computing dq is slightly different. we make N_CTX // BLOCK_N2 passes over dq
    # For a fixed K/V block, each contributes something to dq
    start_m = pid_0 * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    dq = _attn_bwd_dq(
        dq, q, K, V,
        do, m, DELTA,
        stride_tok, stride_d,
        H, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps,
        MASK=False,
    )

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)

class _attn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # Get the head dim of query and key tensors
        HDIM_q = q.shape[-1]
        HDIM_k = k.shape[-1]
        HDIM_v = v.shape[-1]
        assert HDIM_q == HDIM_k and HDIM_k == HDIM_v
        assert HDIM_k in {16, 32, 64, 128, 256}

        # Initialize attn output 
        o = torch.empty_like(q)
        stage = 3 if causal else 1

        # Initialize 3D empty tensor of shape (batch_size, num_heads, ctx_len) to store LSM statistics for each token
        # M stores one scalar/token per batch x head combination, useful for backward pass without 
        M = torch.empty(q.shape[0], q.shape[1], 1, dtype=torch.float32, device=q.device)

        # Launch 2D grid of block instances
        def grid(META):
            return (triton.cdiv(q.shape[2], META['BLOCK_M']), q.shape[0] * q.shape[1], 1)

        # call the attention forward function
        _attn_fwd[grid](sm_scale, M, 
                        q.shape[0], q.shape[1],   # batch size, head count
                        q, k, v, o,
                        N_CTX = q.shape[2],
                        HEAD_DIM = HDIM_k,
                        STAGE = stage,
                        )
        ctx.grid = grid
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HDIM_k
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        HEAD_DIM = ctx.HEAD_DIM
        assert do.is_contiguous()
        # tuple of integers representing no of steps in memory needed to move to the next element across each dim
        assert q.stride() == k.stride() == v.stride() == o.stride()   # a tensor of shape (2, 8, 256, 32) has strides of (524288,8192,32,1)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, CTX_SIZE = q.shape[:3]
        PRE_BLOCK = 128
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (sm_scale * RCP_LN2)
        assert CTX_SIZE % PRE_BLOCK == 0
        pre_grid = (CTX_SIZE//PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_backward_preprocess[pre_grid](
            o, do, 
            delta, 
            BATCH, N_HEAD, CTX_SIZE,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM
        )
        grid = (CTX_SIZE//BLOCK_N1, BATCH * N_HEAD)
        _attn_backward[grid](
            q, arg_k, sm_scale, v, do, dq, dk, dv, 
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, CTX_SIZE,
            BLOCK_M1 = BLOCK_M1, BLOCK_N1 = BLOCK_N1,
            BLOCK_M2 = BLOCK_M2, BLOCK_N2 = BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=HEAD_DIM,
            CAUSAL=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None

attention = _attn.apply
