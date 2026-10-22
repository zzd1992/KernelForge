from functools import lru_cache

import tilelang
import tilelang.language as T
import torch

from .original import LiteLA as OriginalLiteLA


THREADS = 128
ATTN_BLOCK_CONFIGS = (32, 64, 128, 256)
ATTN_THREADS = (128, 256)
QKV_BLOCK_M_CONFIGS = (64, 128)
QKV_BLOCK_N_CONFIGS = (128,)
QKV_BLOCK_K_CONFIGS = (32, 64)
TORCH_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float",
}


@tilelang.autotune(
    configs=[
        {"block_m": block_m, "block_n": block_n, "block_k": block_k, "threads": threads}
        for block_m in QKV_BLOCK_M_CONFIGS
        for block_n in QKV_BLOCK_N_CONFIGS
        for block_k in QKV_BLOCK_K_CONFIGS
        for threads in ATTN_THREADS
    ],
    warmup=10,
    rep=50,
)
@tilelang.jit(out_idx=[-1], target="cuda")
def _qkv_projection(
    batch: int,
    seq_len: int,
    channels: int,
    dtype: str,
    block_m: int = None,
    block_n: int = None,
    block_k: int = None,
    threads: int = None,
):
    @T.prim_func
    def main(
        X: T.Tensor((batch, seq_len, channels), dtype),
        WeightT: T.Tensor((channels, 3 * channels), dtype),
        Output: T.Tensor((batch, seq_len, 3 * channels), dtype),
    ):
        with T.Kernel(
            batch, T.ceildiv(seq_len, block_m), T.ceildiv(3 * channels, block_n), threads=threads
        ) as (batch_idx, m_tile, n_tile):
            x_shared = T.alloc_shared((block_m, block_k), dtype)
            weight_shared = T.alloc_shared((block_k, block_n), dtype)
            output_local = T.alloc_fragment((block_m, block_n), "float")

            T.clear(output_local)
            for k_tile in T.Pipelined(T.ceildiv(channels, block_k), num_stages=2):
                for row, col in T.Parallel(block_m, block_k):
                    token = m_tile * block_m + row
                    channel = k_tile * block_k + col
                    x_shared[row, col] = 0.0
                    if token < seq_len and channel < channels:
                        x_shared[row, col] = X[batch_idx, token, channel]
                for row, col in T.Parallel(block_k, block_n):
                    channel = k_tile * block_k + row
                    output_channel = n_tile * block_n + col
                    weight_shared[row, col] = 0.0
                    if channel < channels and output_channel < 3 * channels:
                        weight_shared[row, col] = WeightT[channel, output_channel]
                T.gemm(x_shared, weight_shared, output_local)

            for row, col in T.Parallel(block_m, block_n):
                token = m_tile * block_m + row
                output_channel = n_tile * block_n + col
                if token < seq_len and output_channel < 3 * channels:
                    Output[batch_idx, token, output_channel] = T.Cast(dtype, output_local[row, col])

    return main


def qkv_projection(batch: int, seq_len: int, channels: int, dtype: str = "bfloat16"):
    """Return a TileLang no-bias QKV projection expecting packed WeightT `[C, 3C]`."""
    return _qkv_projection(batch, seq_len, channels, dtype)


@tilelang.autotune(
    configs=[
        {"block_k": block_k, "threads": threads}
        for block_k in ATTN_BLOCK_CONFIGS
        for threads in ATTN_THREADS
    ],
    warmup=10,
    rep=50,
)
@tilelang.jit(out_idx=[-2, -1], target="cuda")
def _qkv_vk(
    batch: int,
    heads: int,
    dim: int,
    seq_len: int,
    splits: int,
    dtype: str,
    block_k: int = None,
    threads: int = None,
):
    rows_per_col = threads // dim
    split_len = T.ceildiv(seq_len, splits)

    @T.prim_func
    def main(
        QKV: T.Tensor((batch, seq_len, 3, heads * dim), dtype),
        VK: T.Tensor((batch, heads, splits, dim, dim), dtype),
        KSum: T.Tensor((batch, heads, splits, dim), dtype),
    ):
        with T.Kernel(batch, heads, splits, threads=threads) as (
            batch_idx,
            head_idx,
            split_idx,
        ):
            thread_idx = T.get_thread_binding(0)
            v_shared = T.alloc_shared((dim, block_k), dtype)
            k_shared = T.alloc_shared((block_k, dim), dtype)
            sum_shared = T.alloc_shared((threads,), "float")
            vk_local = T.alloc_fragment((dim, dim), "float")
            k_sum = T.alloc_local((1,), "float")
            partial_sum = T.alloc_local((1,), "float")

            T.clear(vk_local)
            k_sum[0] = 0.0
            for k_tile in T.Pipelined(T.ceildiv(split_len, block_k), num_stages=2):
                for row, col in T.Parallel(dim, block_k):
                    token = split_idx * split_len + k_tile * block_k + col
                    v_shared[row, col] = 0.0
                    if token < seq_len:
                        v_shared[row, col] = QKV[batch_idx, token, 2, head_idx * dim + row]
                for row, col in T.Parallel(block_k, dim):
                    token = split_idx * split_len + k_tile * block_k + row
                    k_shared[row, col] = 0.0
                    if token < seq_len:
                        k_shared[row, col] = T.max(
                            QKV[batch_idx, token, 1, head_idx * dim + col], 0.0
                        )

                partial_sum[0] = 0.0
                if thread_idx < rows_per_col * dim:
                    for row_group in T.Unroll(T.ceildiv(block_k, rows_per_col)):
                        row = thread_idx // dim + row_group * rows_per_col
                        if row < block_k:
                            partial_sum[0] += T.Cast("float", k_shared[row, thread_idx % dim])
                sum_shared[thread_idx] = partial_sum[0]
                T.sync_threads()
                if thread_idx < dim:
                    for worker in T.Unroll(rows_per_col):
                        k_sum[0] += sum_shared[worker * dim + thread_idx]

                T.gemm(v_shared, k_shared, vk_local)
                T.sync_threads()

            T.copy(vk_local, VK[batch_idx, head_idx, split_idx, 0, 0])
            if thread_idx < dim:
                KSum[batch_idx, head_idx, split_idx, thread_idx] = T.Cast(dtype, k_sum[0])

    return main


@tilelang.autotune(
    configs=[
        {"block_n": block_n, "threads": threads}
        for block_n in ATTN_BLOCK_CONFIGS
        for threads in ATTN_THREADS
    ],
    warmup=10,
    rep=50,
)
@tilelang.jit(out_idx=[-1], target="cuda")
def _qkv_out(
    batch: int,
    heads: int,
    dim: int,
    seq_len: int,
    splits: int,
    eps: float,
    dtype: str,
    output_dtype: str,
    block_n: int = None,
    threads: int = None,
):
    @T.prim_func
    def main(
        QKV: T.Tensor((batch, seq_len, 3, heads * dim), dtype),
        VK: T.Tensor((batch, heads, splits, dim, dim), dtype),
        KSum: T.Tensor((batch, heads, splits, dim), dtype),
        Output: T.Tensor((batch, seq_len, heads * dim), output_dtype),
    ):
        with T.Kernel(batch, heads, T.ceildiv(seq_len, block_n), threads=threads) as (
            batch_idx,
            head_idx,
            seq_tile,
        ):
            thread_idx = T.get_thread_binding(0)
            q_shared = T.alloc_shared((dim, block_n), dtype)
            vk_shared = T.alloc_shared((dim, dim), dtype)
            denominator = T.alloc_shared((block_n,), "float")
            out_local = T.alloc_fragment((dim, block_n), "float")

            for row, col in T.Parallel(dim, block_n):
                token = seq_tile * block_n + col
                q_shared[row, col] = 0.0
                if token < seq_len:
                    q_shared[row, col] = T.max(
                        QKV[batch_idx, token, 0, head_idx * dim + row], 0.0
                    )

            T.clear(out_local)
            for split_idx in T.Serial(splits):
                T.copy(VK[batch_idx, head_idx, split_idx, 0, 0], vk_shared)
                T.gemm(vk_shared, q_shared, out_local)

            for col in T.Parallel(block_n):
                denominator[col] = 0.0
                for split_idx in T.Serial(splits):
                    for row in T.Serial(dim):
                        denominator[col] += T.Cast(
                            "float", KSum[batch_idx, head_idx, split_idx, row]
                        ) * T.Cast("float", q_shared[row, col])

            for row, col in T.Parallel(dim, block_n):
                if seq_tile * block_n + col < seq_len:
                    Output[batch_idx, seq_tile * block_n + col, head_idx * dim + row] = T.Cast(
                        output_dtype, out_local[row, col] / (denominator[col] + eps)
                    )

    return main


@tilelang.jit(out_idx=[-2, -1], target="cuda")
def _qkv_reduce(
    batch: int,
    heads: int,
    dim: int,
    splits: int,
    dtype: str,
):
    @T.prim_func
    def main(
        VK: T.Tensor((batch, heads, splits, dim, dim), dtype),
        KSum: T.Tensor((batch, heads, splits, dim), dtype),
        ReducedVK: T.Tensor((batch, heads, dim, dim), dtype),
        ReducedKSum: T.Tensor((batch, heads, dim), "float"),
    ):
        with T.Kernel(batch, heads, threads=THREADS) as (batch_idx, head_idx):
            vk_sum = T.alloc_local((1,), "float")
            k_sum = T.alloc_local((1,), "float")

            for row, col in T.Parallel(dim, dim):
                vk_sum[0] = 0.0
                for split_idx in T.Serial(splits):
                    vk_sum[0] += T.Cast("float", VK[batch_idx, head_idx, split_idx, row, col])
                ReducedVK[batch_idx, head_idx, row, col] = T.Cast(dtype, vk_sum[0])

            for row in T.Parallel(dim):
                k_sum[0] = 0.0
                for split_idx in T.Serial(splits):
                    k_sum[0] += T.Cast("float", KSum[batch_idx, head_idx, split_idx, row])
                ReducedKSum[batch_idx, head_idx, row] = k_sum[0]

    return main


@tilelang.autotune(
    configs=[
        {"block_n": block_n, "threads": threads}
        for block_n in ATTN_BLOCK_CONFIGS
        for threads in ATTN_THREADS
    ],
    warmup=10,
    rep=50,
)
@tilelang.jit(out_idx=[-1], target="cuda")
def _qkv_out_reduced(
    batch: int,
    heads: int,
    dim: int,
    seq_len: int,
    eps: float,
    dtype: str,
    output_dtype: str,
    block_n: int = None,
    threads: int = None,
):
    @T.prim_func
    def main(
        QKV: T.Tensor((batch, seq_len, 3, heads * dim), dtype),
        VK: T.Tensor((batch, heads, dim, dim), dtype),
        KSum: T.Tensor((batch, heads, dim), "float"),
        Output: T.Tensor((batch, seq_len, heads * dim), output_dtype),
    ):
        with T.Kernel(batch, heads, T.ceildiv(seq_len, block_n), threads=threads) as (
            batch_idx,
            head_idx,
            seq_tile,
        ):
            thread_idx = T.get_thread_binding(0)
            q_shared = T.alloc_shared((dim, block_n), dtype)
            vk_shared = T.alloc_shared((dim, dim), dtype)
            k_sum_shared = T.alloc_shared((dim,), "float")
            denominator = T.alloc_shared((block_n,), "float")
            out_local = T.alloc_fragment((dim, block_n), "float")
            denominator_partial = T.alloc_local((1,), "float")

            for row, col in T.Parallel(dim, block_n):
                token = seq_tile * block_n + col
                q_shared[row, col] = 0.0
                if token < seq_len:
                    q_shared[row, col] = T.max(
                        QKV[batch_idx, token, 0, head_idx * dim + row], 0.0
                    )
            T.clear(out_local)
            T.copy(VK[batch_idx, head_idx, 0, 0], vk_shared)
            for row in T.Parallel(dim):
                k_sum_shared[row] = KSum[batch_idx, head_idx, row]
            T.gemm(vk_shared, q_shared, out_local)

            for col in T.Parallel(block_n):
                denominator[col] = 0.0
                for row in T.Serial(dim):
                    denominator[col] += k_sum_shared[row] * T.Cast("float", q_shared[row, col])

            for row, col in T.Parallel(dim, block_n):
                if seq_tile * block_n + col < seq_len:
                    Output[batch_idx, seq_tile * block_n + col, head_idx * dim + row] = T.Cast(
                        output_dtype, out_local[row, col] / (denominator[col] + eps)
                    )

    return main


def qkv_attn_matmul(
    batch: int,
    heads: int,
    dim: int,
    seq_len: int,
    eps: float = 1e-15,
    dtype: str = "bfloat16",
    output_dtype: str = "float",
    reduce_splits: bool = True,
):
    if dim <= 0 or dim > THREADS or THREADS % dim != 0:
        raise ValueError(f"dim must divide {THREADS} and be at most {THREADS}")
    if output_dtype not in ("float", "float16", "bfloat16"):
        raise ValueError("output_dtype must be float, float16, or bfloat16")

    if seq_len >= 16384:
        splits = 16
    elif seq_len >= 2048:
        # The supported shapes need enough producer CTAs to occupy the GPU.
        target_ctas = 256 if heads >= 64 else 128
        splits = min(
            (seq_len + 255) // 256,
            (target_ctas + batch * heads - 1) // (batch * heads),
        )
    else:
        splits = min(
            (seq_len + 255) // 256,
            (64 + batch * heads - 1) // (batch * heads),
        )
    build_vk = _qkv_vk(batch, heads, dim, seq_len, splits, dtype)
    reduce_partials = reduce_splits and seq_len >= 2048 and splits > 1
    if reduce_partials:
        reduce_partials_kernel = _qkv_reduce(batch, heads, dim, splits, dtype)
        build_reduced_out = _qkv_out_reduced(batch, heads, dim, seq_len, eps, dtype, output_dtype)

        def run(qkv):
            vk, k_sum = build_vk(qkv)
            vk, k_sum = reduce_partials_kernel(vk, k_sum)
            return build_reduced_out(qkv, vk, k_sum)

        return run

    build_out = _qkv_out(batch, heads, dim, seq_len, splits, eps, dtype, output_dtype)

    def run(qkv):
        vk, k_sum = build_vk(qkv)
        return build_out(qkv, vk, k_sum)

    return run


class LiteLA(OriginalLiteLA):
    """LiteLA with a packed-QKV TileLang fast path."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        heads=None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-15,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
    ):
        if dtype not in TORCH_DTYPES:
            raise ValueError(f"unsupported attention dtype: {dtype}")
        if qk_norm:
            raise ValueError("Custom LiteLA supports only qk_norm=False")
        super().__init__(
            in_dim, out_dim, heads, heads_ratio, dim, eps, use_bias, qk_norm, norm_eps
        )
        self.input_shape = (batch, seq_len, in_dim)
        self.dtype = dtype
        self.qkv_projection_kernel = None
        if not use_bias:
            self.qkv_projection_kernel = qkv_projection(
                batch, seq_len, in_dim, TORCH_DTYPES[dtype]
            )
            self.register_buffer(
                "qkv_weight_t", self.qkv.weight.detach().t().contiguous(), persistent=False
            )
        self.qkv_kernel = qkv_attn_matmul(
            batch, self.heads, self.dim, seq_len, eps, TORCH_DTYPES[dtype], TORCH_DTYPES[dtype]
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.qkv_projection_kernel is not None:
            self.qkv_weight_t.copy_(self.qkv.weight.t())
        return result

    def _validate_input(self, x):
        if x.shape != self.input_shape:
            raise ValueError(f"expected LiteLA input shape {self.input_shape}, got {tuple(x.shape)}")
        if x.dtype != self.dtype:
            raise ValueError(f"expected LiteLA dtype {self.dtype}, got {x.dtype}")

    def forward(self, x, mask=None, HW=None, rotary_emb=None, block_id=None, block_mask=None):
        if rotary_emb is not None or getattr(self, "fp32_attention", False):
            return super().forward(x, mask, HW, rotary_emb, block_id, block_mask)
        self._validate_input(x)

        batch, seq_len, channels = x.shape
        if self.qkv_projection_kernel is None:
            qkv = self.qkv(x).reshape(batch, seq_len, 3, channels)
        else:
            qkv = self.qkv_projection_kernel(x, self.qkv_weight_t).reshape(batch, seq_len, 3, channels)
        out = self.qkv_kernel(qkv).to(x.dtype)
        return self.proj(out)


@tilelang.autotune(
    configs=[
        {"block_m": block_m, "block_n": block_n, "block_k": block_k, "threads": threads}
        for block_m in (64, 128)
        for block_n in (64, 128)
        for block_k in (32, 64)
        for threads in ATTN_THREADS
    ],
    warmup=10,
    rep=50,
)
@tilelang.jit(out_idx=[-1], target="cuda")
def _output_projection(
    batch: int,
    seq_len: int,
    channels: int,
    dtype: str,
    block_m: int = None,
    block_n: int = None,
    block_k: int = None,
    threads: int = None,
):
    """Bias-fused output projection using prepacked WeightT `[C,C]`."""

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq_len, channels), dtype),
        WeightT: T.Tensor((channels, channels), dtype),
        Bias: T.Tensor((channels,), dtype),
        Output: T.Tensor((batch, seq_len, channels), dtype),
    ):
        with T.Kernel(
            batch,
            T.ceildiv(seq_len, block_m),
            T.ceildiv(channels, block_n),
            threads=threads,
        ) as (batch_idx, m_tile, n_tile):
            x_shared = T.alloc_shared((block_m, block_k), dtype)
            weight_shared = T.alloc_shared((block_k, block_n), dtype)
            output_local = T.alloc_fragment((block_m, block_n), "float")

            T.clear(output_local)
            for k_tile in T.Pipelined(T.ceildiv(channels, block_k), num_stages=2):
                for row, col in T.Parallel(block_m, block_k):
                    token = m_tile * block_m + row
                    channel = k_tile * block_k + col
                    x_shared[row, col] = 0.0
                    if token < seq_len and channel < channels:
                        x_shared[row, col] = X[batch_idx, token, channel]
                for row, col in T.Parallel(block_k, block_n):
                    channel = k_tile * block_k + row
                    output_channel = n_tile * block_n + col
                    weight_shared[row, col] = 0.0
                    if channel < channels and output_channel < channels:
                        weight_shared[row, col] = WeightT[channel, output_channel]
                T.gemm(x_shared, weight_shared, output_local)

            for row, col in T.Parallel(block_m, block_n):
                token = m_tile * block_m + row
                output_channel = n_tile * block_n + col
                if token < seq_len and output_channel < channels:
                    Output[batch_idx, token, output_channel] = T.Cast(
                        dtype,
                        output_local[row, col] + T.Cast("float", Bias[output_channel]),
                    )

    return main


class JointLiteLA(LiteLA):
    """Standard SANA LiteLA with custom QKV, attention, and output kernels."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        heads=None,
        heads_ratio: float = 1.0,
        dim: int = 32,
        eps: float = 1e-15,
        use_bias: bool = False,
        qk_norm: bool = False,
        norm_eps: float = 1e-5,
    ):
        if in_dim != out_dim:
            raise ValueError("JointLiteLA requires in_dim == out_dim")
        super().__init__(
            in_dim,
            out_dim,
            batch=batch,
            seq_len=seq_len,
            dtype=dtype,
            heads=heads,
            heads_ratio=heads_ratio,
            dim=dim,
            eps=eps,
            use_bias=use_bias,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.output_projection_kernel = _output_projection(
            batch, seq_len, out_dim, TORCH_DTYPES[dtype]
        )
        self.register_buffer(
            "proj_weight_t", self.proj.weight.detach().t().contiguous(), persistent=False
        )
        if self.proj.bias is None:
            self.register_buffer(
                "proj_bias", torch.zeros(out_dim, device=self.proj.weight.device, dtype=dtype),
                persistent=False,
            )
        else:
            self.register_buffer("proj_bias", self.proj.bias.detach().clone(), persistent=False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.proj_weight_t.copy_(self.proj.weight.t())
        if self.proj.bias is not None:
            self.proj_bias.copy_(self.proj.bias)
        return result

    def forward(self, x, mask=None, HW=None, rotary_emb=None, block_id=None, block_mask=None):
        if rotary_emb is not None or getattr(self, "fp32_attention", False):
            return super().forward(x, mask, HW, rotary_emb, block_id, block_mask)
        self._validate_input(x)

        batch, seq_len, channels = x.shape
        qkv = self.qkv_projection_kernel(x, self.qkv_weight_t).reshape(
            batch, seq_len, 3, channels
        )
        out = self.qkv_kernel(qkv).to(x.dtype)
        return self.output_projection_kernel(out, self.proj_weight_t, self.proj_bias)


GROUPED_BATCH = 1
GROUPED_SEQ_LEN = 1024
GROUPED_CHANNELS = 1152
GROUPED_HEADS = 36
GROUPED_DIM = 32
GROUPED_SPLITS = 16
GROUPED_DTYPE = "bfloat16"
GROUPED_PRODUCER_HEAD_GROUP = 2
GROUPED_QOUT_CONFIG = (2, 128, 64, 256)


@lru_cache(maxsize=1)
def grouped_full64_kv_vk_producer(head_group=GROUPED_PRODUCER_HEAD_GROUP):
    """Build the selected full-64, serial group-2 producer."""
    if head_group != GROUPED_PRODUCER_HEAD_GROUP:
        raise ValueError("only the selected producer head group 2 is retained")
    block_m = 64
    block_k = 32
    threads = 128
    group_width = head_group * GROUPED_DIM

    @tilelang.jit(out_idx=[-2, -1], target="cuda")
    def build_group2():
        @T.prim_func
        def group2_main(
            X: T.Tensor((GROUPED_BATCH, GROUPED_SEQ_LEN, GROUPED_CHANNELS), GROUPED_DTYPE),
            WeightT: T.Tensor((GROUPED_CHANNELS, 3 * GROUPED_CHANNELS), GROUPED_DTYPE),
            VK: T.Tensor(
                (GROUPED_BATCH, GROUPED_HEADS, GROUPED_SPLITS, GROUPED_DIM, GROUPED_DIM),
                GROUPED_DTYPE,
            ),
            KSum: T.Tensor(
                (GROUPED_BATCH, GROUPED_HEADS, GROUPED_SPLITS, GROUPED_DIM), GROUPED_DTYPE
            ),
        ):
            with T.Kernel(
                GROUPED_BATCH,
                GROUPED_HEADS // head_group,
                GROUPED_SPLITS,
                threads=threads,
            ) as (batch_idx, group_idx, split_idx):
                thread_idx = T.get_thread_binding(0)
                x_shared = T.alloc_shared((block_m, block_k), GROUPED_DTYPE)
                weight_shared = T.alloc_shared((block_k, group_width), GROUPED_DTYPE)
                grouped_k = T.alloc_shared((block_m, group_width), GROUPED_DTYPE)
                grouped_v = T.alloc_shared((group_width, block_m), GROUPED_DTYPE)
                projected_k = T.alloc_fragment((block_m, group_width), "float")
                projected_v = T.alloc_fragment((block_m, group_width), "float")
                vk_local = T.alloc_fragment((GROUPED_DIM, GROUPED_DIM), "float")
                ksum = T.alloc_local((1,), "float")

                T.clear(projected_k)
                T.clear(projected_v)
                for channel_tile in T.Serial(GROUPED_CHANNELS // block_k):
                    for row, col in T.Parallel(block_m, block_k):
                        x_shared[row, col] = X[
                            batch_idx,
                            split_idx * block_m + row,
                            channel_tile * block_k + col,
                        ]
                    for row, col in T.Parallel(block_k, group_width):
                        weight_shared[row, col] = WeightT[
                            channel_tile * block_k + row,
                            GROUPED_CHANNELS + group_idx * group_width + col,
                        ]
                    T.gemm(x_shared, weight_shared, projected_k)
                    T.sync_threads()
                    for row, col in T.Parallel(block_k, group_width):
                        weight_shared[row, col] = WeightT[
                            channel_tile * block_k + row,
                            2 * GROUPED_CHANNELS + group_idx * group_width + col,
                        ]
                    T.gemm(x_shared, weight_shared, projected_v)
                    T.sync_threads()

                for row, col in T.Parallel(block_m, group_width):
                    grouped_k[row, col] = T.Cast(
                        GROUPED_DTYPE, T.max(projected_k[row, col], 0.0)
                    )
                    grouped_v[col, row] = T.Cast(GROUPED_DTYPE, projected_v[row, col])
                T.sync_threads()

                for group_head in T.Serial(head_group):
                    head = group_idx * head_group + group_head
                    T.clear(vk_local)
                    ksum[0] = 0.0
                    if thread_idx < GROUPED_DIM:
                        for row in T.Serial(block_m):
                            ksum[0] += T.Cast(
                                "float", grouped_k[row, group_head * GROUPED_DIM + thread_idx]
                            )
                    T.gemm(
                        grouped_v[
                            group_head * GROUPED_DIM : (group_head + 1) * GROUPED_DIM,
                            0:block_m,
                        ],
                        grouped_k[
                            0:block_m,
                            group_head * GROUPED_DIM : (group_head + 1) * GROUPED_DIM,
                        ],
                        vk_local,
                    )
                    T.copy(vk_local, VK[batch_idx, head, split_idx, 0, 0])
                    if thread_idx < GROUPED_DIM:
                        KSum[batch_idx, head, split_idx, thread_idx] = T.Cast(
                            GROUPED_DTYPE, ksum[0]
                        )
                    T.sync_threads()

        return group2_main

    return build_group2()


@tilelang.jit(out_idx=[-1], target="cuda")
def partial_reduce_joint_qout():
    """Reduce producer partials inside the selected group-2 Q/output CTA."""
    head_group, token_tile, k_tile, threads = GROUPED_QOUT_CONFIG
    group_width = head_group * GROUPED_DIM

    @T.prim_func
    def main(
        X: T.Tensor((GROUPED_BATCH, GROUPED_SEQ_LEN, GROUPED_CHANNELS), GROUPED_DTYPE),
        WqT: T.Tensor((GROUPED_CHANNELS, GROUPED_CHANNELS), GROUPED_DTYPE),
        VK: T.Tensor(
            (GROUPED_BATCH, GROUPED_HEADS, GROUPED_SPLITS, GROUPED_DIM, GROUPED_DIM),
            GROUPED_DTYPE,
        ),
        KSum: T.Tensor(
            (GROUPED_BATCH, GROUPED_HEADS, GROUPED_SPLITS, GROUPED_DIM), GROUPED_DTYPE
        ),
        Output: T.Tensor(
            (GROUPED_BATCH, GROUPED_SEQ_LEN, GROUPED_CHANNELS), GROUPED_DTYPE
        ),
    ):
        with T.Kernel(
            GROUPED_BATCH,
            GROUPED_SEQ_LEN // token_tile,
            GROUPED_HEADS // head_group,
            threads=threads,
        ) as (batch_idx, token_block, head_block):
            x_shared = T.alloc_shared((token_tile, k_tile), GROUPED_DTYPE)
            weight_shared = T.alloc_shared((k_tile, group_width), GROUPED_DTYPE)
            q_shared = T.alloc_shared((group_width, token_tile), GROUPED_DTYPE)
            vk_shared = T.alloc_shared((GROUPED_DIM, GROUPED_DIM), GROUPED_DTYPE)
            ksum_shared = T.alloc_shared((GROUPED_DIM,), "float")
            denominator = T.alloc_shared((token_tile,), "float")
            q_accum = T.alloc_fragment((token_tile, group_width), "float")
            out_accum = T.alloc_fragment((GROUPED_DIM, token_tile), "float")
            vk_sum = T.alloc_local((1,), "float")
            ksum_sum = T.alloc_local((1,), "float")

            T.clear(q_accum)
            for input_block in T.Pipelined(GROUPED_CHANNELS // k_tile, num_stages=2):
                for row, col in T.Parallel(token_tile, k_tile):
                    x_shared[row, col] = X[
                        batch_idx,
                        token_block * token_tile + row,
                        input_block * k_tile + col,
                    ]
                for row, col in T.Parallel(k_tile, group_width):
                    weight_shared[row, col] = WqT[
                        input_block * k_tile + row,
                        head_block * group_width + col,
                    ]
                T.gemm(x_shared, weight_shared, q_accum)

            for row, col in T.Parallel(token_tile, group_width):
                q_shared[col, row] = T.max(T.Cast(GROUPED_DTYPE, q_accum[row, col]), 0.0)
            T.sync_threads()

            for group_head in T.Serial(head_group):
                head = head_block * head_group + group_head
                for row, col in T.Parallel(GROUPED_DIM, GROUPED_DIM):
                    vk_sum[0] = 0.0
                    for split_idx in T.Serial(GROUPED_SPLITS):
                        vk_sum[0] += T.Cast(
                            "float", VK[batch_idx, head, split_idx, row, col]
                        )
                    vk_shared[row, col] = T.Cast(GROUPED_DTYPE, vk_sum[0])
                for row in T.Parallel(GROUPED_DIM):
                    ksum_sum[0] = 0.0
                    for split_idx in T.Serial(GROUPED_SPLITS):
                        ksum_sum[0] += T.Cast(
                            "float", KSum[batch_idx, head, split_idx, row]
                        )
                    ksum_shared[row] = ksum_sum[0]
                T.sync_threads()

                T.clear(out_accum)
                T.gemm(
                    vk_shared,
                    q_shared[
                        group_head * GROUPED_DIM : (group_head + 1) * GROUPED_DIM,
                        0:token_tile,
                    ],
                    out_accum,
                )
                for col in T.Parallel(token_tile):
                    denominator[col] = 0.0
                    for row in T.Serial(GROUPED_DIM):
                        denominator[col] += ksum_shared[row] * T.Cast(
                            "float", q_shared[group_head * GROUPED_DIM + row, col]
                        )
                for row, col in T.Parallel(GROUPED_DIM, token_tile):
                    Output[
                        batch_idx,
                        token_block * token_tile + col,
                        head * GROUPED_DIM + row,
                    ] = T.Cast(
                        GROUPED_DTYPE, out_accum[row, col] / (denominator[col] + 1e-15)
                    )
                T.sync_threads()

    return main


class GroupedFull64QOutLiteLA(JointLiteLA):
    """Selected full-64 producer plus partial-reducing joint Q/output."""

    def __init__(self, *args, **kwargs):
        batch = kwargs.get("batch")
        seq_len = kwargs.get("seq_len")
        dtype = kwargs.get("dtype")
        dim = kwargs.get("dim", GROUPED_DIM)
        qk_norm = kwargs.get("qk_norm", False)
        use_bias = kwargs.get("use_bias", False)
        in_dim = args[0] if args else kwargs.get("in_dim")
        out_dim = args[1] if len(args) > 1 else kwargs.get("out_dim")
        heads = kwargs.get("heads")
        if (
            (batch, seq_len, in_dim, out_dim)
            != (GROUPED_BATCH, GROUPED_SEQ_LEN, GROUPED_CHANNELS, GROUPED_CHANNELS)
            or (heads is not None and heads != GROUPED_HEADS)
            or dtype != torch.bfloat16
            or dim != GROUPED_DIM
            or qk_norm
            or use_bias
        ):
            raise ValueError(
                "GroupedFull64QOutLiteLA requires exactly B=1,N=1024,C=1152,H=36,D=32, "
                "bf16, qk_norm=False, and use_bias=False"
            )
        super().__init__(*args, **kwargs)
        self.grouped_kv_producer_kernel = grouped_full64_kv_vk_producer()
        self.partial_reduce_qout_kernel = partial_reduce_joint_qout()
        self.register_buffer(
            "q_weight_t",
            self.qkv.weight[:GROUPED_CHANNELS].detach().t().contiguous(),
            persistent=False,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.q_weight_t.copy_(self.qkv.weight[:GROUPED_CHANNELS].t())
        return result

    def grouped_attention(self, x):
        vk, ksum = self.grouped_kv_producer_kernel(x, self.qkv_weight_t)
        return self.partial_reduce_qout_kernel(x, self.q_weight_t, vk, ksum)

    def forward(self, x, mask=None, HW=None, rotary_emb=None, block_id=None, block_mask=None):
        if rotary_emb is not None:
            raise ValueError("GroupedFull64QOutLiteLA requires rotary_emb=None")
        self._validate_input(x)
        out = self.grouped_attention(x)
        return self.output_projection_kernel(out, self.proj_weight_t, self.proj_bias)


def optimized_joint_litela(in_dim: int, out_dim: int, **kwargs):
    """Build the fastest retained module for a target shape."""
    shape = (kwargs.get("batch"), kwargs.get("seq_len"), in_dim)
    module_cls = GroupedFull64QOutLiteLA if shape == (1, 1024, 1152) else JointLiteLA
    return module_cls(in_dim, out_dim, **kwargs)


@lru_cache(maxsize=None)
def _joint_pipeline(batch, seq_len, channels, heads, dim, eps, dtype):
    project_qkv = qkv_projection(batch, seq_len, channels, dtype)
    attention = qkv_attn_matmul(batch, heads, dim, seq_len, eps, dtype, dtype)
    project_output = _output_projection(batch, seq_len, channels, dtype)

    def run(x, qkv_weight_t, proj_weight_t, proj_bias):
        qkv = project_qkv(x, qkv_weight_t).reshape(batch, seq_len, 3, channels)
        out = attention(qkv)
        return project_output(out, proj_weight_t, proj_bias)

    return run


@torch.library.custom_op("kernelforge::joint_litela_forward", mutates_args=())
def _joint_litela_forward(
    x: torch.Tensor,
    qkv_weight_t: torch.Tensor,
    proj_weight_t: torch.Tensor,
    proj_bias: torch.Tensor,
    heads: int,
    dim: int,
    eps: float,
) -> torch.Tensor:
    dtype = str(x.dtype).removeprefix("torch.")
    pipeline = _joint_pipeline(x.shape[0], x.shape[1], x.shape[2], heads, dim, eps, dtype)
    return pipeline(x, qkv_weight_t, proj_weight_t, proj_bias)


@_joint_litela_forward.register_fake
def _joint_litela_forward_fake(x, qkv_weight_t, proj_weight_t, proj_bias, heads, dim, eps):
    return x.new_empty((*x.shape[:-1], proj_weight_t.shape[-1]))


class CustomOpJointLiteLA(JointLiteLA):
    """JointLiteLA exposed to Dynamo as one opaque custom operation."""

    def forward(self, x, mask=None, HW=None, rotary_emb=None, block_id=None, block_mask=None):
        if rotary_emb is not None or getattr(self, "fp32_attention", False):
            return super().forward(x, mask, HW, rotary_emb, block_id, block_mask)
        self._validate_input(x)
        return _joint_litela_forward(
            x,
            self.qkv_weight_t,
            self.proj_weight_t,
            self.proj_bias,
            self.heads,
            self.dim,
            self.eps,
        )


class CUDAGraphJointLiteLA:
    """Fixed-shape CUDA Graph replay wrapper with input-copy cost included."""

    def __init__(self, module: JointLiteLA):
        self.module = module
        self.static_input = torch.empty(
            module.input_shape, device=module.proj_weight_t.device, dtype=module.dtype
        )
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                module(self.static_input)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_output = module(self.static_input)

    def __call__(self, x):
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output


__all__ = [
    "CUDAGraphJointLiteLA",
    "CustomOpJointLiteLA",
    "GroupedFull64QOutLiteLA",
    "JointLiteLA",
    "LiteLA",
    "grouped_full64_kv_vk_producer",
    "optimized_joint_litela",
    "partial_reduce_joint_qout",
    "qkv_attn_matmul",
    "qkv_projection",
]
