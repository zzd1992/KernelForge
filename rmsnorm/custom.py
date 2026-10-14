import tilelang
import tilelang.language as T
import torch
import torch.nn as nn


THREADS = 128
WARP_SIZE = 32


@tilelang.jit(
    out_idx=[-1],
    target="cuda",
)
def rms_norm(
    batch: int,
    seq_len: int,
    dim: int,
    eps: float = 1e-6,
    dtype: str = "bfloat16",
):
    values_per_thread = (dim + THREADS - 1) // THREADS
    num_warps = THREADS // WARP_SIZE

    @T.prim_func
    def main(
        X: T.Tensor((batch, seq_len, dim), dtype),
        Weight: T.Tensor((dim,), dtype),
        Output: T.Tensor((batch, seq_len, dim), dtype),
    ):
        with T.Kernel(batch * seq_len, threads=THREADS) as row:
            thread_idx = T.get_thread_binding(0)
            x_local = T.alloc_local((values_per_thread,), "float")
            thread_sum = T.alloc_local((1,), "float")
            block_sum = T.alloc_local((1,), "float")
            warp_sums = T.alloc_shared((num_warps,), "float")

            batch_idx = row // seq_len
            seq_idx = row % seq_len
            thread_sum[0] = 0.0

            for i in T.Unroll(values_per_thread):
                col = i * THREADS + thread_idx
                x_local[i] = 0.0
                if col < dim:
                    x_local[i] = T.Cast("float", X[batch_idx, seq_idx, col])
                    thread_sum[0] += x_local[i] * x_local[i]

            thread_sum[0] = T.warp_reduce_sum(thread_sum[0])
            if thread_idx % WARP_SIZE == 0:
                warp_sums[thread_idx // WARP_SIZE] = thread_sum[0]
            T.sync_threads()

            if thread_idx < WARP_SIZE:
                block_sum[0] = 0.0
                if thread_idx < num_warps:
                    block_sum[0] = warp_sums[thread_idx]
                block_sum[0] = T.warp_reduce_sum(block_sum[0])
                if thread_idx == 0:
                    warp_sums[0] = block_sum[0]
            T.sync_threads()

            for i in T.Unroll(values_per_thread):
                col = i * THREADS + thread_idx
                if col < dim:
                    x_local[i] = T.Cast(
                        "float",
                        T.Cast(
                            dtype,
                            x_local[i] * T.rsqrt(warp_sums[0] / dim + eps),
                        ),
                    ) * T.Cast("float", Weight[col])
                    Output[batch_idx, seq_idx, col] = T.Cast(dtype, x_local[i])

    return main
