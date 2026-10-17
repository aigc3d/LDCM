# Robust Poisson Depth Completion

## Method

Building on Eq. (4) in the
[paper](https://arxiv.org/abs/2605.30115), we retain the paper's notation:
$\mathbf{D}^{\mathrm{prior}}$ denotes the dense coarse depth prior,
$\mathbf{D}^{\mathrm{mono}}$ the initial monocular relative depth prediction,
$\tilde{\mathbf{D}}$ the filtered sparse depth, and
$\mathcal{M}^{\mathrm{valid}}$ the set of valid sparse observations.

For the implementation below, let
$\bar{\mathbf{D}}^{\mathrm{mono}}=a\mathbf{D}^{\mathrm{mono}}+b$ denote the
affine-aligned monocular prediction. At each pyramid level, the solver minimizes
the following robust objective:

$$
\min_{\mathbf{D}^{\mathrm{prior}}>0}
\left[\sum_{(p,q)\in\mathcal{E}} w_{pq}
\left(\log\mathbf{D}^{\mathrm{prior}}_q-\log\mathbf{D}^{\mathrm{prior}}_p-g_{pq}\right)^2
+\lambda\sum_{p\in\mathcal{M}^{\mathrm{valid}}}\rho_p
\left(\log\mathbf{D}^{\mathrm{prior}}_p-\log\tilde{\mathbf{D}}_p\right)^2\right]
$$

Here, $\mathcal{E}$ is the set of horizontally and vertically adjacent pixel
pairs, and $g_{pq}$ is the clipped log-depth difference induced by the aligned
monocular prediction:

$$
g_{pq}=\mathrm{clip}\left(
\log\bar{\mathbf{D}}^{\mathrm{mono}}_q-\log\bar{\mathbf{D}}^{\mathrm{mono}}_p,-3,3
\right).
$$

Unlike the compact formulation in Eq. (4), the updated implementation evaluates
the sparse-data term in log-depth space. It also introduces the following
edge-aware weights:

$$
w_{pq}=\max\left(10^{-4},\min(c_p,c_q)\max\left(0.01,
\exp\left[-\frac{
|\log\bar{\mathbf{D}}^{\mathrm{mono}}_q-\log\bar{\mathbf{D}}^{\mathrm{mono}}_p|
}{0.01}\right]\right)\right)
$$

Here, $c$ denotes the optional confidence and defaults to one when confidence is
not provided. For directional confidence, the horizontal or vertical channel is
used for the corresponding edge. Consequently, large monocular depth
discontinuities weaken the coupling across likely object boundaries.

For each usable sparse observation, define
$r_p=\log\tilde{\mathbf{D}}_p-\log\bar{\mathbf{D}}^{\mathrm{mono}}_p$.
An adaptive local window estimates the neighboring correction mean $\mu_p$ and
standard deviation $\sigma_p$, excluding the center observation itself. The
observation reliability is then

$$
\rho_p=\max\left(0.05,
\frac{1}{1+\left(\frac{r_p-\mu_p}{\max(0.15,2\sigma_p)}\right)^2}\right).
$$

The window size adapts to the per-image sparse-observation density and targets
approximately 12 usable observations. If fewer than three usable neighbors are
available, the reliability remains one. These reliability weights are computed
once, without a learned module or an outer robust-optimization loop. At coarser
pyramid levels, sparse depths are pooled using the reliability weights, and
reliability is averaged over occupied pixels. A conjugate-gradient solver then
proceeds from coarse to fine before exponentiating the log-depth solution and
resizing it to the input resolution.

## Acknowledgment

We thank the authors of
[MetricScenes](https://github.com/MetricScenes/MetricScenes), whose edge-aware
Poisson completion implementation inspired our edge-weighting design.
