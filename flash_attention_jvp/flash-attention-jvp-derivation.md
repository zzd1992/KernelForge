# FlashAttention JVP: From the Mathematical Definition to an Online Tiled Algorithm

This document derives the Jacobian-vector product (JVP) of standard softmax attention and reformulates it as an online tiled algorithm compatible with FlashAttention. The goal is to compute both the attention output and its directional derivative in a single scan over the key/value blocks, without explicitly constructing the Jacobian or storing the full attention matrix.

All identities below hold in exact real arithmetic. GPU implementations are subject to floating-point rounding, accumulation precision, and the order in which tiles are processed.

## 1. Problem Setup and Notation

Consider one batch element and one attention head. To cover both self-attention and cross-attention, let the query and key sequence lengths be $N_q$ and $N_k$, respectively:

$$
\begin{aligned}
Q &\in \mathbb{R}^{N_q\times d},\\
K &\in \mathbb{R}^{N_k\times d},\\
V &\in \mathbb{R}^{N_k\times d_v}.
\end{aligned}
$$

With the fixed scale $\gamma=1/\sqrt{d}$, standard attention is defined by

$$
\begin{aligned}
S &= \gamma QK^\top,\\
P &= \operatorname{softmax}_{\mathrm{row}}(S),\\
O &= PV.
\end{aligned}
\tag{1}
$$

Here, $S,P\in\mathbb{R}^{N_q\times N_k}$ and $O\in\mathbb{R}^{N_q\times d_v}$.

The derivation assumes:

- The scale $\gamma$ is fixed.
- Dropout is disabled.
- Any attention mask is fixed, and every query row has at least one valid key.
- Masked positions are excluded from the sums. In an implementation, their exponential weights are set to zero.
- A fixed, finite additive bias may be included in $S$ without changing the derivative formulas. If the bias also varies along the chosen direction, its tangent must be added to $\dot{S}$.

Unless otherwise stated, sums and maxima over keys are restricted to valid positions.

| Symbol | Meaning |
| --- | --- |
| $\dot{Q},\dot{K},\dot{V}$ | Input tangents, with the same shapes as their corresponding primal values |
| $\dot{S},\dot{P},\dot{O}$ | Directional derivatives of intermediate quantities and the output |
| $\odot$ | Elementwise multiplication, with row-wise broadcasting where specified |
| $\operatorname{rowsum}(A)$ | Sum over each row of $A$, retaining a singleton final dimension |
| $\operatorname{rowmax}(A)$ | Maximum over each row of $A$, retaining a singleton final dimension |
| $\mathbf{1}$ | An all-ones column vector of the appropriate dimension |

A dot denotes a directional derivative, not a reverse-mode loss gradient. A row-wise scalar statistic is represented as a column vector and broadcast across the columns of the corresponding matrix.

## 2. Definition of the JVP

For a given direction $(\dot{Q},\dot{K},\dot{V})$, define the input path

$$
\begin{aligned}
Q(\varepsilon) &= Q+\varepsilon\dot{Q},\\
K(\varepsilon) &= K+\varepsilon\dot{K},\\
V(\varepsilon) &= V+\varepsilon\dot{V}.
\end{aligned}
$$

The desired JVP is

$$
\boxed{
\dot{O}
=
\left.
\frac{d}{d\varepsilon}
\operatorname{Attention}
\bigl(Q(\varepsilon),K(\varepsilon),V(\varepsilon)\bigr)
\right|_{\varepsilon=0}.
}
\tag{2}
$$

If the three inputs are flattened and concatenated into a vector $z$, Equation (2) becomes

$$
\operatorname{vec}(\dot{O})=J(z)\dot{z}.
$$

Computing the JVP requires only the action of the Jacobian on a direction vector, not the full matrix $J(z)$.

## 3. Deriving the JVP of Standard Attention

### 3.1 Directional Derivative of the Score Matrix

Expand the matrix product:

$$
\begin{aligned}
S(\varepsilon)
&=\gamma(Q+\varepsilon\dot{Q})
          (K+\varepsilon\dot{K})^\top\\
&=\gamma QK^\top
  +\varepsilon\gamma
   \bigl(\dot{Q}K^\top+Q\dot{K}^\top\bigr)\\
&\quad+\varepsilon^2\gamma\dot{Q}\dot{K}^\top.
\end{aligned}
$$

Taking the first derivative at $\varepsilon=0$ gives

$$
\boxed{
\dot{S}
=
\gamma\bigl(\dot{Q}K^\top+Q\dot{K}^\top\bigr).
}
\tag{3}
$$

The term $\dot{Q}\dot{K}^\top$ contributes only at second order in $\varepsilon$ and therefore does not appear in the JVP.

### 3.2 Directional Derivative of Softmax

For a fixed query row $i$, write

$$
p_{ij}=\frac{e^{s_{ij}}}{Z_i},
\qquad
Z_i=\sum_k e^{s_{ik}}.
$$

Using the derivative of the exponential and the quotient rule,

$$
\begin{aligned}
\dot{p}_{ij}
&=
\frac{e^{s_{ij}}\dot{s}_{ij}}{Z_i}
-
\frac{e^{s_{ij}}}{Z_i^2}
\sum_k e^{s_{ik}}\dot{s}_{ik}\\
&=
p_{ij}\dot{s}_{ij}
-
p_{ij}\sum_k p_{ik}\dot{s}_{ik}.
\end{aligned}
$$

Define the row-wise statistic

$$
\begin{aligned}
c_i &= \sum_j p_{ij}\dot{s}_{ij},\\
c &= \operatorname{rowsum}(P\odot\dot{S})
     \in\mathbb{R}^{N_q\times 1}.
\end{aligned}
$$

Then

$$
\boxed{
\begin{aligned}
\dot{p}_{ij}
&=p_{ij}(\dot{s}_{ij}-c_i),\\
\dot{P}
&=P\odot\bigl(\dot{S}-c\mathbf{1}^\top\bigr).
\end{aligned}
}
\tag{4}
$$

The scalar $c_i$ is the attention-weighted mean of the score perturbation in row $i$. Subtracting it ensures that the probability perturbations sum to zero:

$$
\sum_j\dot{p}_{ij}=0.
$$

### 3.3 Directional Derivative of the Output

Since $O=PV$, the product rule gives

$$
\dot{O}=\dot{P}V+P\dot{V}.
$$

Substituting Equation (4),

$$
\begin{aligned}
\dot{O}
&=
\bigl[P\odot(\dot{S}-c\mathbf{1}^\top)\bigr]V
+P\dot{V}\\
&=
(P\odot\dot{S})V
-\bigl[P\odot(c\mathbf{1}^\top)\bigr]V
+P\dot{V}\\
&=
(P\odot\dot{S})V-c\odot(PV)+P\dot{V}.
\end{aligned}
$$

Therefore,

$$
\boxed{
\begin{aligned}
\dot{O}
&=P\dot{V}+(P\odot\dot{S})V-c\odot O,\\
c&=\operatorname{rowsum}(P\odot\dot{S}).
\end{aligned}
}
\tag{5}
$$

The three contributions have distinct meanings:

1. $P\dot{V}$ accounts for changes in the values.
2. $(P\odot\dot{S})V$ accounts for score-dependent changes in the weighted sum.
3. $-c\odot O$ corrects for the change in the softmax denominator.

For a single query row, the same identity is

$$
\boxed{
\dot{o}_i
=
\sum_j p_{ij}\dot{v}_j
+
\sum_j p_{ij}\dot{s}_{ij}(v_j-o_i).
}
\tag{6}
$$

Equation (5) is the starting point for the FlashAttention JVP algorithm.

## 4. Why Independent Softmax Computations per Tile Are Insufficient

FlashAttention processes $Q,K,V$ in tiles to avoid writing the full $S$ and $P$ matrices to device memory.

However, softmax normalizes over all valid keys in a query row:

$$
p_{ij}
=
\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}.
$$

Applying softmax independently within each key tile and then adding the tile outputs would use the wrong normalization denominators.

The JVP has another dependency on the full row:

$$
c_i=\sum_j p_{ij}\dot{s}_{ij}.
$$

While processing a tile, neither the final denominator nor $c_i$ is known yet. Equation (5) must therefore be expressed using quantities that can be accumulated tile by tile and normalized after the scan.

## 5. Unnormalized Accumulators for the Output and Its JVP

Fix one query and omit the row index $i$. For the set $\mathcal{J}$ of valid keys processed so far, define

$$
m=\max_{j\in\mathcal{J}}s_j,
\qquad
e_j=e^{s_j-m}.
$$

Maintain the following four weighted sums:

$$
\begin{aligned}
\ell &= \sum_{j\in\mathcal{J}}e_j,\\
u &= \sum_{j\in\mathcal{J}}e_jv_j,\\
r &= \sum_{j\in\mathcal{J}}e_j\dot{s}_j,\\
g &= \sum_{j\in\mathcal{J}}e_j
     \bigl(\dot{s}_jv_j+\dot{v}_j\bigr).
\end{aligned}
\tag{7}
$$

Together with $m$, these form five running states.

| State | Shape per query | Meaning |
| --- | --- | --- |
| $m$ | Scalar | Running maximum score, used as a stable exponential reference |
| $\ell$ | Scalar | Sum of unnormalized weights |
| $u$ | Vector of length $d_v$ | Unnormalized output numerator |
| $r$ | Scalar | Unnormalized weighted sum of score perturbations |
| $g$ | Vector of length $d_v$ | Unnormalized sum of the two positive JVP terms in Equation (5) |

After all keys have been processed,

$$
p_j=\frac{e_j}{\ell},
\qquad
o=\frac{u}{\ell},
\qquad
c=\frac{r}{\ell}.
$$

Moreover,

$$
\frac{g}{\ell}
=
\sum_j p_j\dot{s}_jv_j
+
\sum_j p_j\dot{v}_j.
$$

Substituting into Equation (5) gives

$$
\boxed{
\begin{aligned}
o&=\frac{u}{\ell},\\
\dot{o}&=\frac{g}{\ell}-\frac{r}{\ell}\,o.
\end{aligned}
}
\tag{8}
$$

Thus, adding the two accumulators $r,g$ to the usual FlashAttention states $m,\ell,u$ is sufficient to recover the JVP at the end of the scan.

The quantities $r,g$ are defined as weighted sums. Because the reference $m$ can also vary, they are not directly equal to $\dot{\ell},\dot{u}$.

## 6. Why the Derivative of the Running Maximum Is Unnecessary

### 6.1 Direct Cancellation

At a point where $m$ is differentiable,

$$
\dot{e}_j=e_j(\dot{s}_j-\dot{m}).
$$

Consequently,

$$
\begin{aligned}
\dot{\ell}
&=\sum_j e_j(\dot{s}_j-\dot{m})\\
&=r-\dot{m}\,\ell,\\[4pt]
\dot{u}
&=\sum_j
  \bigl[e_j(\dot{s}_j-\dot{m})v_j+e_j\dot{v}_j\bigr]\\
&=g-\dot{m}\,u.
\end{aligned}
$$

Differentiating $o=u/\ell$,

$$
\begin{aligned}
\dot{o}
&=\frac{\dot{u}}{\ell}
  -\frac{u\dot{\ell}}{\ell^2}\\
&=\frac{g-\dot{m}\,u}{\ell}
  -\frac{u(r-\dot{m}\,\ell)}{\ell^2}\\
&=\frac{g}{\ell}-\frac{ur}{\ell^2}.
\end{aligned}
$$

Hence,

$$
\boxed{
\dot{o}=\frac{g}{\ell}-\frac{r}{\ell}\,o.
}
\tag{9}
$$

Every term involving $\dot{m}$ cancels. The kernel therefore does not need to compute or maintain $\dot{m}$.

### 6.2 Ties in the Maximum

The derivative of attention does not require a unique maximum score. For any finite constant $a$,

$$
\frac{\sum_j e^{s_j-a}v_j}{\sum_j e^{s_j-a}}
=
\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.
$$

At the differentiation point, choose $a$ to equal the maximum score at that point, then hold $a$ fixed while differentiating. The two expressions are identical functions of the original inputs, so they have identical derivatives.

The running maximum is a common shift chosen for numerical stability. Differentiating the max operation is unnecessary, even when multiple entries attain the maximum.

## 7. Online Tiled Recurrences

Let query tile $I$ contain $B_q$ positions and key/value tile $J$ contain $B_k$ positions.

Keep $Q_I,\dot{Q}_I$ fixed while scanning the tiles $K_J,V_J,\dot{K}_J,\dot{V}_J$. Maintain independent states for each query row:

$$
\begin{aligned}
m,\ell,r &\in\mathbb{R}^{B_q\times 1},\\
U,G &\in\mathbb{R}^{B_q\times d_v}.
\end{aligned}
$$

### 7.1 Initialization

$$
m=-\infty,\qquad
\ell=0,\qquad
U=0,\qquad
r=0,\qquad
G=0.
\tag{10}
$$

### 7.2 Tile Scores and Score Tangents

$$
\boxed{
\begin{aligned}
S_{IJ}
&=\gamma Q_IK_J^\top,\\
\dot{S}_{IJ}
&=\gamma
  \bigl(\dot{Q}_IK_J^\top+Q_I\dot{K}_J^\top\bigr).
\end{aligned}
}
\tag{11}
$$

Both matrices have shape $B_q\times B_k$. With a fixed mask, invalid entries are excluded from the following maxima and weighted sums.

### 7.3 Updating the Exponential Reference

$$
\begin{aligned}
m_{\mathrm{new}}
&=\max\bigl(m,\operatorname{rowmax}(S_{IJ})\bigr),\\
\alpha&=e^{m-m_{\mathrm{new}}}.
\end{aligned}
\tag{12}
$$

The maximum and exponential are evaluated independently for each query row. Section 8 provides the guard needed when a row has not yet encountered any valid key.

For a previously processed key $j$,

$$
\begin{aligned}
e^{s_j-m_{\mathrm{new}}}
&=e^{m-m_{\mathrm{new}}}e^{s_j-m}\\
&=\alpha e^{s_j-m}.
\end{aligned}
$$

All previous weighted sums must therefore be multiplied by the same row-wise factor $\alpha$.

### 7.4 Tile Weights and Tangent-Weighted Scores

$$
\boxed{
\begin{aligned}
E&=\exp\bigl(S_{IJ}-m_{\mathrm{new}}\mathbf{1}^\top\bigr),\\
H&=E\odot\dot{S}_{IJ}.
\end{aligned}
}
\tag{13}
$$

Here, $E,H\in\mathbb{R}^{B_q\times B_k}$.

$E$ contains unnormalized weights relative to the current exponential reference. It is not a probability matrix normalized independently within the tile. Likewise, $H$ is not the final $\dot{P}$: the normalization correction has not yet been applied.

### 7.5 Updating the Accumulators

$$
\boxed{
\begin{aligned}
\ell_{\mathrm{new}}
&=\alpha\odot\ell+\operatorname{rowsum}(E),\\
U_{\mathrm{new}}
&=\alpha\odot U+EV_J,\\
r_{\mathrm{new}}
&=\alpha\odot r+\operatorname{rowsum}(H),\\
G_{\mathrm{new}}
&=\alpha\odot G+HV_J+E\dot{V}_J.
\end{aligned}
}
\tag{14}
$$

Replace the states with their updated values and continue to the next key/value tile.

**Invariant proof.** After each update, $\ell,U,r,G$ equal the weighted sums in Equation (7) over all valid keys processed so far, using the updated reference $m_{\mathrm{new}}$. Initialization represents the empty set. Each update rescales the contribution of the old set and adds the contribution of the new tile. By induction, the final states equal the weighted sums over the entire row.

### 7.6 Final Normalization

$$
\boxed{
\begin{aligned}
O_I&=\frac{U}{\ell},\\
\dot{O}_I
&=\frac{G}{\ell}-\frac{r}{\ell}\odot O_I.
\end{aligned}
}
\tag{15}
$$

Every division is broadcast row-wise. The assumption of at least one valid key per query guarantees $\ell>0$ at the end.

If the log-sum-exp statistic is also needed, compute

$$
\operatorname{LSE}_I=m+\log\ell.
$$

Equations (11)-(15) compute the forward output and its JVP in one scan over the key/value tiles.

## 8. Pseudocode and Mask Edge Cases

The following pseudocode processes one query tile. Row statistics retain a singleton final dimension. Without masking, every entry of the validity mask is true.

~~~text
Inputs:
    QI, tQI
    Key/value tiles KJ, VJ, tKJ, tVJ
    Fixed scale gamma
    Fixed validity mask, restricted to each tile as valid

Initialize:
    m   = -inf           # [Bq, 1]
    ell = 0              # [Bq, 1]
    U   = 0              # [Bq, dv]
    r   = 0              # [Bq, 1]
    G   = 0              # [Bq, dv]

for each key/value tile J:
    S  = gamma * (QI @ transpose(KJ))
    tS = gamma * (tQI @ transpose(KJ) + QI @ transpose(tKJ))

    S = where(valid, S, -inf)
    m_new = maximum(m, rowmax(S))

    # If a row has seen no valid key, m_new is -inf.
    # A finite reference avoids evaluating -inf - (-inf).
    m_ref = where(isfinite(m_new), m_new, 0)

    alpha = exp(m - m_ref)
    E = exp(S - m_ref)
    H = E * where(valid, tS, 0)

    ell = alpha * ell + rowsum(E)
    U   = alpha * U   + E @ VJ
    r   = alpha * r   + rowsum(H)
    G   = alpha * G   + H @ VJ + E @ tVJ
    m   = m_new

O  = U / ell
tO = G / ell - (r / ell) * O

Return O, tO
~~~

The guard handles two cases:

- The current tile is entirely invalid for a row, but an earlier tile contained a valid key. The previous states remain unchanged.
- No valid key has been encountered for a row so far. All four weighted sums remain zero until the first valid key appears.

If a row has no valid key after the entire scan, standard softmax is undefined for that row. Its output semantics must be specified separately; this case is outside the assumptions of this document.

The derivation also assumes finite logits and tangents at valid positions. Infinities caused by FP16/BF16 overflow are not resolved by the mask guard.

## 9. Computational and Memory Costs

### 9.1 Matrix Multiplication Work

| Stage | Forward only | Forward + JVP |
| --- | --- | --- |
| Scores and score tangents | $QK^\top$ | $QK^\top,\ \dot{Q}K^\top,\ Q\dot{K}^\top$ |
| Output and tangent accumulation | $EV$ | $EV,\ HV,\ E\dot{V}$ |
| Equivalent full matrix products | 2 | 6 |

These counts describe matrix multiplication work, not the number of GPU kernel launches.

Counting each multiplication and addition as one FLOP, and ignoring softmax, scaling, and elementwise operations,

$$
\begin{aligned}
\operatorname{FLOPs}_{\mathrm{forward}}
&\approx 2N_qN_k(d+d_v),\\
\operatorname{FLOPs}_{\mathrm{forward+JVP}}
&\approx 6N_qN_k(d+d_v).
\end{aligned}
$$

For general nonzero tangents, the dominant matrix multiplication FLOPs of fused forward + JVP are approximately three times those of forward alone. Adding JVP to the same forward scan contributes approximately twice the matrix multiplication FLOPs of forward alone.

This ratio is not a fixed latency multiplier. Actual performance also depends on bandwidth, register usage, tile sizes, Tensor Core utilization, and overlap between operations.

### 9.2 Memory Requirements

The full matrices $S,P,\dot{S},\dot{P}$ need not be written to device memory. Scores, exponential weights, and tangent-weighted scores exist only as tile-local intermediates.

For one query tile, the main on-chip working storage scales as

$$
O\bigl(
B_qB_k+(B_q+B_k)d+(B_q+B_k)d_v
\bigr).
$$

Primal/tangent pairs and multiple temporary tile buffers affect the constant factors.

For fixed head dimensions, the total tensor storage for the inputs, input tangents, outputs, and output tangents grows linearly with sequence length. No $N_qN_k$ attention-weight matrix is stored. This is a statement about tensor storage, not a claim that total HBM traffic is necessarily linear.

### 9.3 Special Cases

If $\dot{Q}=\dot{K}=0$, then $\dot{S}=0$ and

$$
\dot{O}=P\dot{V}.
$$

If $\dot{V}=0$, the term $E\dot{V}_J$ can be omitted.

Such optimizations require the caller or kernel to know that the relevant tangents are zero. Small tangent magnitudes alone do not justify dropping their contributions.

## 10. Input Tangents versus Reverse-Mode Gradients

The inputs $\dot{Q},\dot{K},\dot{V}$ to FlashAttention JVP are rates of change along a specified direction, not loss gradients.

For example, suppose

$$
\begin{aligned}
Q&=XW_Q+b_Q,\\
K&=XW_K+b_K,\\
V&=XW_V+b_V.
\end{aligned}
$$

With fixed parameters and input tangent $\dot{X}$,

$$
\begin{aligned}
\dot{Q}&=\dot{X}W_Q,\\
\dot{K}&=\dot{X}W_K,\\
\dot{V}&=\dot{X}W_V.
\end{aligned}
$$

Forward-mode automatic differentiation propagates these tangents from earlier layers along the same direction.

| Operation | Inputs | Outputs |
| --- | --- | --- |
| Attention forward | $Q,K,V$ | $O$ |
| Attention JVP | Primal values and $\dot{Q},\dot{K},\dot{V}$ | $\dot{O}$ |
| Attention VJP / backward | Primal values and $\nabla_O L$ | $\nabla_Q L,\nabla_K L,\nabla_V L$ |

This document derives forward + JVP. If a training objective also requires backpropagation through $\dot{O}$, the implementation needs a backward rule for the JVP computation. The tangent formulas alone do not establish that a kernel supports that capability.

## 11. Consistency Checks

### 11.1 Probability Mass Is Preserved

From Equation (4),

$$
\begin{aligned}
\sum_j\dot{p}_{ij}
&=\sum_j p_{ij}\dot{s}_{ij}
  -c_i\sum_j p_{ij}\\
&=c_i-c_i=0.
\end{aligned}
$$

### 11.2 A Uniform Score Shift Leaves the Output Unchanged

If $\dot{s}_{ij}=a_i$ for every valid key in row $i$, then $c_i=a_i$ and $\dot{P}=0$.

If $\dot{V}=0$ as well,

$$
\dot{O}=0.
$$

### 11.3 Value-Only Perturbations Reduce to a Linear Weighted Sum

When $\dot{Q}=\dot{K}=0$, we have $H=0$ and $r=0$. Equation (15) reduces directly to

$$
\dot{O}=P\dot{V}.
$$

### 11.4 Checking an Implementation with Central Finite Differences

Let $F(Q,K,V)=\operatorname{Attention}(Q,K,V)$ and define

$$
\begin{aligned}
F_+(\varepsilon)
&=F\bigl(Q+\varepsilon\dot{Q},
         K+\varepsilon\dot{K},
         V+\varepsilon\dot{V}\bigr),\\
F_-(\varepsilon)
&=F\bigl(Q-\varepsilon\dot{Q},
         K-\varepsilon\dot{K},
         V-\varepsilon\dot{V}\bigr).
\end{aligned}
$$

With a fixed mask and deterministic computation,

$$
\dot{O}
\approx
\frac{F_+(\varepsilon)-F_-(\varepsilon)}{2\varepsilon}.
$$

For a smooth function, the central-difference truncation error is $O(\varepsilon^2)$. Numerically, choosing $\varepsilon$ too small increases rounding and cancellation errors. Start with small FP64 matrices when checking the formulas, and compare several tile sizes. Use tolerances appropriate to the precision and scale when validating a low-precision GPU kernel.

## 12. References

The equations above are derived directly from the definition of standard attention. The following sources provide the tiling background, an example JVP kernel, and the automatic differentiation interfaces.

1. Tri Dao et al. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (2022).  
   [Paper](https://arxiv.org/abs/2205.14135)

2. Tri Dao. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (2023).  
   [Paper](https://arxiv.org/abs/2307.08691)

3. NVIDIA NVlabs/rcm. **FlashAttention JVP Triton implementation**. Primal values and tangents are passed as q/k/v and tq/tk/tv, respectively, and the function returns the output and its tangent. An implementation may accumulate the two contributions to $G$ separately without changing the mathematics.  
   [Source code](https://github.com/NVlabs/rcm/blob/main/rcm/utils/flash_attention_jvp_triton.py)

4. PyTorch. **Extending torch.func with autograd.Function**.  
   [Official documentation](https://docs.pytorch.org/docs/2.8/notes/extending.func.html)

5. PyTorch. **Forward-mode Automatic Differentiation**.  
   [Official tutorial](https://docs.pytorch.org/tutorials/intermediate/forward_ad_usage.html)

> Document date: 2026-11-12. The source-code link points to a mutable branch; use a specific commit when checking implementation interfaces and supported features.

