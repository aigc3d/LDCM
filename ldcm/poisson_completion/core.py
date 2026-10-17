"""Robust depth completion: local sparse reliability and a Poisson pyramid.

Only the validated local-Cauchy configuration is implemented. Gradient guidance
comes from the aligned monocular depth; no training or reference GT is needed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ._solver import poisson_solver


def _finite_mono(mono):
    """Replace invalid values with the positive per-image median (or one)."""
    good = torch.isfinite(mono) & (mono > 0)
    med = torch.nanmedian(torch.where(good, mono, torch.nan).flatten(1), dim=1).values
    med = torch.where(torch.isfinite(med) & (med > 0), med, torch.ones_like(med))
    return torch.where(good, mono, med[:, None, None, None]).clamp_min(1e-6)


def _log(value):
    return torch.where(value > 0, torch.log(value), torch.zeros_like(value))


def _resize(value, size):
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


def _align_mono(sparse, mono):
    """Affine least-squares alignment using only the supplied sparse points."""
    batch_size = sparse.shape[0]
    mask = sparse > 0
    all_x = [mono[i][mask[i]] for i in range(batch_size)]
    all_y = [sparse[i][mask[i]] for i in range(batch_size)]
    counts = [x.numel() for x in all_x]
    valid_indices = [i for i, count in enumerate(counts) if count >= 2]
    if not valid_indices:
        return mono
    max_len = max(counts)
    padded_A_list = []
    padded_b_list = []
    for i in valid_indices:
        x, y = all_x[i], all_y[i]
        num_points = x.numel()
        A = torch.stack([x, torch.ones_like(x)], dim=1)
        b = y.unsqueeze(1)
        pad_len = max_len - num_points
        padded_A_list.append(F.pad(A, (0, 0, 0, pad_len)))
        padded_b_list.append(F.pad(b, (0, 0, 0, pad_len)))
    A_batch = torch.stack(padded_A_list, dim=0)
    b_batch = torch.stack(padded_b_list, dim=0)
    try:
        solution = torch.linalg.lstsq(A_batch, b_batch).solution
    except torch.linalg.LinAlgError:
        return mono
    scales = solution[:, 0:1, :].squeeze(-1)
    shifts = solution[:, 1:2, :].squeeze(-1)
    aligned_mono = mono.clone()
    final_scales = torch.ones(batch_size, 1, 1, 1, device=mono.device, dtype=mono.dtype)
    final_shifts = torch.zeros(batch_size, 1, 1, 1, device=mono.device, dtype=mono.dtype)
    valid_indices_tensor = torch.tensor(valid_indices, device=mono.device, dtype=torch.long)
    final_scales.index_put_((valid_indices_tensor,), scales.unsqueeze(-1).unsqueeze(-1))
    final_shifts.index_put_((valid_indices_tensor,), shifts.unsqueeze(-1).unsqueeze(-1))
    return F.relu(torch.nan_to_num(aligned_mono * final_scales + final_shifts))


def _box_sum(value, radius):
    """Square-window sums with the scan order of the validated implementation."""
    width = 2 * radius + 1
    rows = F.pad(value, (radius + 1, radius, 0, 0)).cumsum(dim=-1)
    rows = rows[..., width:] - rows[..., :-width]
    columns = F.pad(rows, (0, 0, radius + 1, radius)).cumsum(dim=-2)
    return columns[..., width:, :] - columns[..., :-width, :]


def _anchor_reliability(sparse, aligned_mono):
    """Local Cauchy weights; every valid observation retains positive weight.

    Compare each log correction with neighbors, excluding itself. The window
    targets 12 observations. Fewer than 3 neighbors leaves weight one; otherwise
    q=max(.05, 1/(1+(residual/max(.15,2*std))**2)).
    """
    valid = torch.isfinite(sparse) & (sparse > 0)
    usable = valid & torch.isfinite(aligned_mono) & (aligned_mono > 0)
    work_dtype = torch.float64 if sparse.dtype == torch.float64 else torch.float32
    safe_sparse = torch.where(usable, sparse, torch.ones_like(sparse)).to(work_dtype)
    safe_mono = torch.where(usable, aligned_mono, torch.ones_like(aligned_mono)).to(work_dtype)
    correction = safe_sparse.log() - safe_mono.log()
    weights = valid.to(work_dtype)
    height, width = sparse.shape[-2:]
    for index in range(sparse.shape[0]):
        mask = usable[index:index + 1]
        r = correction[index:index + 1]
        count = int(mask.sum().item())
        if count < 4:
            continue
        expected_radius = round(math.sqrt(height * width * 12.0 / count) / 2)
        max_radius = max(1, min(height, width) // 4)
        radius = min(max_radius, max(4, expected_radius))
        mask_float = mask.to(work_dtype)
        moments = torch.cat((mask_float, mask_float * r, mask_float * r.square()), dim=1)
        neighbor_moments = _box_sum(moments, radius) - moments
        neighbor_count = neighbor_moments[:, 0:1]
        denominator = neighbor_count.clamp_min(1.0)
        center = neighbor_moments[:, 1:2] / denominator
        variance = (neighbor_moments[:, 2:3] / denominator - center.square()).clamp_min(0)
        cutoff = (2.0 * variance.sqrt()).clamp_min(0.15)
        ratio = (r - center).abs() / cutoff.clamp_min(1e-12)
        candidate = (1.0 + ratio.square()).reciprocal().clamp(min=0.05, max=1.0)
        supported = mask & (neighbor_count >= 3)
        weights[index:index + 1] = torch.where(supported, candidate, weights[index:index + 1])
    return weights.to(sparse.dtype)


def _pool_sparse(sparse, reliability, factor):
    """Pool weighted depths and mean reliability per occupied cell."""
    valid = sparse > 0
    weights = torch.where(valid, reliability, torch.zeros_like(reliability))
    if factor == 1:
        return sparse, valid, weights
    counts = F.avg_pool2d(valid.to(sparse.dtype), factor, factor)
    confidence = F.avg_pool2d(weights, factor, factor)
    weighted_depth = F.avg_pool2d(sparse * weights, factor, factor)
    low_valid = counts > 0
    low_sparse = torch.where(low_valid, weighted_depth / confidence.clamp_min(1e-12),
                             torch.zeros_like(weighted_depth))
    low_reliability = torch.where(low_valid, confidence / counts.clamp_min(1e-12),
                                  torch.zeros_like(confidence))
    return low_sparse, low_valid, low_reliability


def _gradient(log_mono, threshold):
    dx = log_mono[:, :, :, 1:] - log_mono[:, :, :, :-1]
    dy = log_mono[:, :, 1:, :] - log_mono[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    gradient = torch.stack([dx, dy], dim=1).squeeze(2)
    return torch.clamp(gradient, min=-threshold, max=threshold)


def _edge_weights(log_mono, confidence, sigma, floor):
    if confidence is None:
        confidence = torch.ones_like(log_mono)
    if confidence.shape[1] == 1:
        confidence = confidence.expand(-1, 2, -1, -1)
    horizontal = torch.minimum(confidence[:, 0:1, :, :-1], confidence[:, 0:1, :, 1:])
    vertical = torch.minimum(confidence[:, 1:2, :-1, :], confidence[:, 1:2, 1:, :])
    dx = (log_mono[:, :, :, 1:] - log_mono[:, :, :, :-1]).abs()
    dy = (log_mono[:, :, 1:, :] - log_mono[:, :, :-1, :]).abs()
    wx = horizontal * torch.exp(-dx / float(sigma)).clamp_min(floor)
    wy = vertical * torch.exp(-dy / float(sigma)).clamp_min(floor)
    return torch.cat([F.pad(wx, (0, 1, 0, 0), value=1),
                      F.pad(wy, (0, 0, 0, 1), value=1)], dim=1).clamp_min(1e-4)


def _validate_inputs(sparse, mono, confidence, scales, caps, ratio, lamda, sigma, floor):
    """Check the public tensor contract and positive solver parameters once."""
    if (sparse.ndim != 4 or sparse.shape != mono.shape or sparse.shape[1] != 1
            or min(sparse.shape) < 1 or sparse.dtype not in (torch.float32, torch.float64)):
        raise ValueError("depth inputs must be float32 or float64 tensors with identical nonempty [B,1,H,W] shapes")
    if sparse.device != mono.device or sparse.dtype != mono.dtype:
        raise ValueError("depth inputs must have the same device and dtype")
    if confidence is not None and (
            confidence.ndim != 4 or confidence.shape[0] != sparse.shape[0]
            or confidence.shape[1] not in (1, 2) or confidence.shape[-2:] != sparse.shape[-2:]
            or confidence.device != sparse.device or confidence.dtype != sparse.dtype):
        raise ValueError("confidence must match depth shape, device and dtype, with 1 or 2 channels")
    if scales < 1 or len(caps) != scales or min(caps) < 1:
        raise ValueError("num_scales must match a positive iteration-cap list")
    if not (0 < ratio <= 1 and 1e-4 <= floor <= 1 and 0 < lamda < math.inf and 0 < sigma < math.inf):
        raise ValueError("require ratio in (0,1], edge_floor in [1e-4,1], and finite positive lamda/edge_sigma")


@torch.no_grad()
def poisson_completion(sparse, mono_depth, num_scales=4, thres=3., lamda=5.,
                       rtol=1e-5, max_iter_per_scale=None, confidence=None,
                       max_resolution_ratio=1., *, edge_sigma=.01, edge_floor=.01):
    """Return completed depth with the same [B,1,H,W] shape as the inputs.

    Zero denotes missing sparse observations. Depth inputs must share floating
    dtype (float32 or float64) and device; CUDA float32 uses optional acceleration. Confidence is
    optional [B,1,H,W] or directional [B,2,H,W]. Images without valid sparse
    points return a finite positive monocular map. Frames converge independently.
    This API is inference only; gradients and sparse observations are unchanged
    by an enclosing autocast context.
    """
    if max_iter_per_scale is None:
        max_iter_per_scale = [5000, 2000, 1000, 500, 250][:num_scales]
    _validate_inputs(sparse, mono_depth, confidence, num_scales, max_iter_per_scale,
                     max_resolution_ratio, lamda, edge_sigma, edge_floor)
    # Independent solves avoid continuing a converged frame while another frame
    # is still iterating. Keep single-image arithmetic identical to the reference.
    options = (num_scales, thres, lamda, rtol, max_iter_per_scale,
               max_resolution_ratio, edge_sigma, edge_floor)
    with torch.autocast(device_type=sparse.device.type, enabled=False):
        if sparse.shape[0] == 1:
            return _single_completion(sparse, mono_depth, confidence, *options)
        return torch.cat([
            _single_completion(sparse[i:i + 1], mono_depth[i:i + 1],
                               None if confidence is None else confidence[i:i + 1],
                               *options)
            for i in range(sparse.shape[0])
        ], dim=0)


def _single_completion(sparse, mono_depth, confidence, num_scales, thres, lamda,
                       rtol, max_iter_per_scale, max_resolution_ratio,
                       edge_sigma, edge_floor):
    height, width = sparse.shape[-2:]
    sparse = torch.where(torch.isfinite(sparse) & (sparse > 0), sparse, torch.zeros_like(sparse))
    mono = _finite_mono(mono_depth)
    if not bool((sparse > 0).any()):
        return mono
    if min(height, width) < 2 ** (num_scales - 1):
        raise ValueError("input dimensions are too small for the configured pyramid")

    aligned = _align_mono(sparse, mono)
    # Reliability uses the raw fit; fitted zeros are repaired only for gradients.
    reliability = _anchor_reliability(sparse, aligned)
    reliability = torch.nan_to_num(reliability, nan=0., posinf=0., neginf=0.).clamp(1e-4, 1.)
    aligned = _finite_mono(aligned)
    if confidence is not None:
        confidence = torch.nan_to_num(confidence, nan=0., posinf=0., neginf=0.).clamp_min(0)
    target_height = int(height * max_resolution_ratio)
    stop = next((s for s in range(num_scales) if height // (2 ** s) <= target_height), num_scales - 1)
    current = None
    for scale in range(num_scales - 1, stop - 1, -1):
        factor = 2 ** scale
        size = (height // factor, width // factor)
        low_mono = _resize(aligned, size)
        low_sparse, low_valid, low_reliability = _pool_sparse(sparse, reliability, factor)
        low_confidence = None if confidence is None else _resize(confidence, size)
        log_mono = _log(low_mono)
        gradient = _gradient(log_mono, thres)
        weights = _edge_weights(log_mono, low_confidence, edge_sigma, edge_floor)
        initial = None if current is None else _resize(current, size)
        current = poisson_solver(
            gradient=gradient, sparse_depth=_log(low_sparse), valid_mask=low_valid,
            gradient_confidence=weights, sparse_confidence=low_reliability,
            init_depth=initial, lamda=lamda, rtol=rtol,
            max_iter=int(max_iter_per_scale[num_scales - 1 - scale]))
    result = current.clamp(-10, 10).exp()
    result = _resize(result, (height, width)) if result.shape[-2:] != (height, width) else result
    return result
