import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import tilelang.language as T


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm(A, blk_m):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_local = T.alloc_fragment((blk_m, N), dtype)
        A_pow_local = T.alloc_fragment((blk_m, N), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_local)
        for i, j in T.Parallel(blk_m, N):
            A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
        for i, j in T.Parallel(blk_m, N):
            A_local[i, j] *= A_powsum[i]
        T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return B

