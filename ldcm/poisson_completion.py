import torch
import torch.nn.functional as F

from .utils import log

def least_square_align_lstsq_vectorized(sparse: torch.Tensor, mono: torch.Tensor):
    """
    Aligns a monocular depth map to a sparse depth map using least squares.

    For each item in the batch, it finds a scale `s` and shift `t` that minimize
    the L2 error: || s * mono[mask] + t - sparse[mask] ||^2, where `mask`
    indicates valid sparse depth points. The aligned map is `relu(s * mono + t)`.

    Args:
        sparse (torch.Tensor): The sparse depth map, with 0 for invalid points.
                               Shape: [batch_size, 1, h, w].
        mono (torch.Tensor): The dense monocular depth map to be aligned.
                             Shape: [batch_size, 1, h, w].

    Returns:
        torch.Tensor: The aligned and rectified monocular depth map.
                      Shape: [batch_size, 1, h, w].
    """
    batch_size = sparse.shape[0]
    mask = sparse > 0

    all_x = [mono[i][mask[i]] for i in range(batch_size)]
    all_y = [sparse[i][mask[i]] for i in range(batch_size)]

    max_len = 0
    valid_indices = []
    for i in range(batch_size):
        num_points = all_x[i].numel()
        if num_points > max_len:
            max_len = num_points
        if num_points >= 2:
            valid_indices.append(i)

    if not valid_indices or max_len == 0:
        return mono

    padded_A_list = []
    padded_b_list = []

    for i in valid_indices:
        x = all_x[i]
        y = all_y[i]
        num_points = x.numel()

        A = torch.stack([x, torch.ones_like(x)], dim=1)
        b = y.unsqueeze(1)

        pad_len = max_len - num_points
        # (padding_left, padding_right, padding_top, padding_bottom)
        A_padded = F.pad(A, (0, 0, 0, pad_len))
        b_padded = F.pad(b, (0, 0, 0, pad_len))

        padded_A_list.append(A_padded)
        padded_b_list.append(b_padded)

    A_batch = torch.stack(padded_A_list, dim=0) # Shape: [num_valid, max_len, 2]
    b_batch = torch.stack(padded_b_list, dim=0) # Shape: [num_valid, max_len, 1]

    try:
        solution = torch.linalg.lstsq(A_batch, b_batch).solution
    except torch.linalg.LinAlgError:
        return mono

    scales = solution[:, 0:1, :].squeeze(-1) # Shape: [num_valid, 1]
    shifts = solution[:, 1:2, :].squeeze(-1) # Shape: [num_valid, 1]

    aligned_mono = mono.clone()
    final_scales = torch.ones(batch_size, 1, 1, 1, device=mono.device, dtype=mono.dtype)
    final_shifts = torch.zeros(batch_size, 1, 1, 1, device=mono.device, dtype=mono.dtype)

    valid_indices_tensor = torch.tensor(valid_indices, device=mono.device, dtype=torch.long)
    final_scales.index_put_((valid_indices_tensor,), scales.unsqueeze(-1).unsqueeze(-1))
    final_shifts.index_put_((valid_indices_tensor,), shifts.unsqueeze(-1).unsqueeze(-1))

    return F.relu(torch.nan_to_num(aligned_mono * final_scales + final_shifts))


def least_square_align(sparse: torch.Tensor, mono: torch.Tensor):
    """
    Aligns a monocular depth map to a sparse depth map using least squares.

    For each item in the batch, it finds a scale `s` and shift `t` that minimize
    the L2 error: || s * mono[mask] + t - sparse[mask] ||^2, where `mask`
    indicates valid sparse depth points. The aligned map is `relu(s * mono + t)`.

    Args:
        sparse (torch.Tensor): The sparse depth map, with 0 for invalid points.
                               Shape: [batch_size, 1, h, w].
        mono (torch.Tensor): The dense monocular depth map to be aligned.
                             Shape: [batch_size, 1, h, w].

    Returns:
        torch.Tensor: The aligned and rectified monocular depth map.
                      Shape: [batch_size, 1, h, w].
    """
    batch_size = sparse.shape[0]
    mask = sparse > 0
    aligned = []
    for i in range(batch_size):
        y = sparse[i][mask[i]]
        x = mono[i][mask[i]]
        if x.numel() < 2: # Not enough points to solve
            aligned.append(mono[i])
            continue
        A = torch.stack([x, torch.ones_like(x)], dim=1)
        try:
            result = torch.linalg.lstsq(A, y.unsqueeze(1))
            sol = result.solution
            scale, shift = sol.squeeze()[0], sol.squeeze()[1]
            aligned.append(F.relu(torch.nan_to_num(mono[i] * scale + shift)))
        except torch.linalg.LinAlgError:
            aligned.append(mono[i]) # Fallback if lstsq fails
    return torch.stack(aligned, dim=0)

class FastFiniteDiffMatrix:
    """
    A matrix-free operator for computing finite differences (gradient) and their
    transpose (divergence) on a batch of images.
    """
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def bmm(self, x: torch.Tensor):
        """
        Computes the forward finite difference (gradient) of a batch of images.

        This is a matrix-free implementation of the matrix-vector product D @ x,
        where D is the finite difference matrix.

        Args:
            x (torch.Tensor): A batch of flattened image vectors.
                              Shape: [batch_size, h * w, 1].

        Returns:
            torch.Tensor: The flattened gradients (dx and dy concatenated).
                          Shape: [batch_size, h*(w-1) + (h-1)*w, 1].
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self.h, self.w)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gradients = torch.cat([dx.reshape(batch_size, -1), dy.reshape(batch_size, -1)], dim=1)
        return gradients.unsqueeze(-1)

    def bmm_transposed(self, x: torch.Tensor):
        """
        Computes the transpose finite difference (divergence) of a batch of gradient fields.

        This is a matrix-free implementation of the matrix-vector product D.T @ y,
        where D.T is the transpose of the finite difference matrix.

        Args:
            x (torch.Tensor): A batch of flattened gradient vectors (dx and dy).
                              Shape: [batch_size, h*(w-1) + (h-1)*w, 1].

        Returns:
            torch.Tensor: The flattened divergence images.
                          Shape: [batch_size, h * w, 1].
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, 1, self.h, self.w, device=x.device, dtype=x.dtype)
        dx_vec = x[:, :self.h * (self.w - 1)]
        dy_vec = x[:, self.h * (self.w - 1):]
        dx = dx_vec.reshape(batch_size, 1, self.h, self.w - 1)
        dy = dy_vec.reshape(batch_size, 1, self.h - 1, self.w)
        output[:, :, :, 1:] += dx
        output[:, :, :, :-1] -= dx
        output[:, :, 1:, :] += dy
        output[:, :, :-1, :] -= dy
        return output.reshape(batch_size, -1, 1)

def batched_matrix_to_flattened(batched_matrix: torch.Tensor):
    """
    Converts a 2-channel gradient image tensor to a flattened vector.

    This prepares the gradient tensor for use with FastFiniteDiffMatrix's transpose operation.

    Args:
        batched_matrix (torch.Tensor): The input gradient tensor with dx and dy channels.
                                       Shape: [batch_size, 2, h, w].

    Returns:
        torch.Tensor: The flattened gradient vector.
                      Shape: [batch_size, h*(w-1) + (h-1)*w, 1].
    """
    batch_size, _, h, w = batched_matrix.shape
    x_truncated = batched_matrix[:, 0, :, :-1].reshape(batch_size, -1)
    y_truncated = batched_matrix[:, 1, :-1, :].reshape(batch_size, -1)
    return torch.cat([x_truncated, y_truncated], dim=-1).unsqueeze(-1)

def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-12, atol=0.0, maxiter=None, early_stopping_improvement_tol=1e-4, early_stopping_consecutive_steps=10):
    """
    Solves a batch of linear systems Ax = B using the Conjugate Gradient method.

    This implementation is matrix-free, meaning it only requires a function
    that computes the matrix-vector product A @ x. It supports batching,
    preconditioning, and early stopping for robustness.

    Args:
        A_bmm (callable): A function that takes a tensor `x` and returns `A @ x`.
        B (torch.Tensor): The right-hand side of the systems. Shape: [batch_size, n, 1].
        M_bmm (callable, optional): A function for the preconditioner's matrix-vector product.
                                    Defaults to the identity matrix.
        X0 (torch.Tensor, optional): The initial guess for the solution X.
                                     Defaults to a zero tensor. Shape: [batch_size, n, 1].
        rtol (float): The relative tolerance for the stopping condition.
        atol (float): The absolute tolerance for the stopping condition.
        maxiter (int, optional): The maximum number of iterations. Defaults to 5 * n.
        early_stopping_improvement_tol (float): Tolerance for relative improvement to decide if
                                               the solver has stalled.
        early_stopping_consecutive_steps (int): Number of consecutive steps with no improvement
                                                before stopping early.

    Returns:
        torch.Tensor: The solution X to the systems Ax = B. Shape: [batch_size, n, 1].
    """
    K, n, m = B.shape
    if M_bmm is None:
        def M_bmm(x): return x
    if X0 is None:
        X0 = torch.zeros_like(B)
    if maxiter is None:
        maxiter = 5 * n

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)
    P_k = Z_k

    B_norm = torch.norm(B, dim=1, keepdim=True)
    stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

    residual_norm = torch.norm(R_k, dim=1, keepdim=True)
    residual_norm_last = residual_norm.clone()

    consecutive_no_improvement = 0

    for k in range(1, maxiter + 1):
        Ap = A_bmm(P_k)
        inner_RZ = torch.sum(R_k * Z_k, dim=1, keepdim=True)
        denominator = torch.sum(P_k * Ap, dim=1, keepdim=True)
        denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=B.device), denominator)
        alpha = inner_RZ / denominator

        X_k = X_k + alpha * P_k
        R_k_new = R_k - alpha * Ap

        residual_norm = torch.norm(R_k_new, dim=1, keepdim=True)

        if (residual_norm <= stopping_matrix).all():
            break

        rel_improvement = (residual_norm_last - residual_norm) / (residual_norm_last + 1e-8)
        if rel_improvement.max() < early_stopping_improvement_tol:
            consecutive_no_improvement += 1
        else:
            consecutive_no_improvement = 0

        if consecutive_no_improvement >= early_stopping_consecutive_steps:
            break

        residual_norm_last = residual_norm.clone()

        Z_k_new = M_bmm(R_k_new)
        beta_denom = torch.where(inner_RZ == 0, torch.tensor(1e-8, device=B.device), inner_RZ)
        beta = torch.sum(R_k_new * Z_k_new, dim=1, keepdim=True) / beta_denom
        P_k = Z_k_new + beta * P_k

        R_k = R_k_new
        Z_k = Z_k_new

    return X_k

def poisson_solver(gradient, sparse_depth, valid_mask, gradient_confidence=None, sparse_confidence=None, init_depth=None, lamda=5.0, rtol=1e-5, max_iter=5000):
    """
    Solves the Poisson equation for single-scale depth completion using CG.

    This function sets up and solves the linear system corresponding to the
    Poisson optimization problem, which aims to find a dense depth map that
    honors both sparse depth measurements and a dense gradient field.

    Args:
        gradient (torch.Tensor): The target dense gradient field, from a mono-depth prediction.
                                 Shape: [batch_size, 2, h, w].
        sparse_depth (torch.Tensor): The sparse depth measurements. Invalid points should be 0.
                                     Shape: [batch_size, 1, h, w].
        valid_mask (torch.Tensor): A boolean/float mask where sparse measurements are valid.
                                   Shape: [batch_size, 1, h, w].
        gradient_confidence (torch.Tensor, optional): Confidence weights for the gradient field.
                                             Shape: [batch_size, 2, h, w]. Defaults to ones.
        sparse_confidence (torch.Tensor, optional): Confidence weights for the sparse depth points.
                                                    Shape: [batch_size, 1, h, w]. Defaults to ones.
        init_depth (torch.Tensor, optional): An initial guess for the dense depth solution.
                                             Shape: [batch_size, 1, h, w]. Defaults to sparse depth.
        lamda (float): The regularization weight balancing the sparse data term and the gradient term.
        rtol (float): Relative tolerance for the CG solver.
        max_iter (int): Maximum number of iterations for the CG solver.

    Returns:
        torch.Tensor: The solved dense depth map. Shape: [batch_size, 1, h, w].
    """
    if gradient_confidence is None:
        gradient_confidence = torch.ones_like(gradient)
    elif gradient_confidence.shape[1] == 1:
        gradient_confidence = gradient_confidence.expand(-1, 2, -1, -1)

    if sparse_confidence is None:
        sparse_confidence = torch.ones_like(sparse_depth)

    batch_size, _, h, w = gradient.shape
    device = gradient.device
    dtype = gradient.dtype

    if not torch.is_tensor(lamda):
        lamda_tensor = torch.tensor(lamda, device=device, dtype=dtype).expand(batch_size, 1, 1)
    else:
        lamda_tensor = lamda

    gradient_confidence = torch.clamp(gradient_confidence, min=1e-4)

    A = FastFiniteDiffMatrix(h, w)

    gradient_flattened = batched_matrix_to_flattened(gradient)
    gradient_confidence_flattened = batched_matrix_to_flattened(gradient_confidence)
    b_bottom = A.bmm_transposed(gradient_confidence_flattened * gradient_flattened)
    b_top = lamda_tensor * (sparse_confidence * valid_mask * sparse_depth).reshape(batch_size, -1, 1)
    b = b_top + b_bottom

    def laplacian_and_mass_matrix_bmm(p):
        grad_term = A.bmm_transposed(gradient_confidence_flattened * A.bmm(p))
        ap = sparse_confidence * valid_mask * p.reshape(batch_size, 1, h, w)
        sparse_term = lamda_tensor * ap.reshape(batch_size, -1, 1)
        return grad_term + sparse_term

    if init_depth is not None:
        init_depth_flat = init_depth.reshape(batch_size, -1, 1)
    else:
        init_depth_flat = (sparse_depth * valid_mask).reshape(batch_size, -1, 1)

    x = cg_batch(laplacian_and_mass_matrix_bmm, b, X0=init_depth_flat, rtol=rtol, maxiter=max_iter)
    return x.reshape(sparse_depth.shape)

def downsample_sparse_depth(sparse_depth: torch.Tensor, scale_factor: int):
    """
    Downsamples a sparse depth map robustly.

    It averages only the valid (non-zero) depth values within each pooling window.
    A downsampled pixel is considered valid if at least one valid pixel from the
    original map contributed to it.

    Args:
        sparse_depth (torch.Tensor): The input sparse depth map.
                                     Shape: [batch_size, 1, h, w].
        scale_factor (int): The factor by which to downsample the height and width.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - avg_pooled_depth (torch.Tensor): The downsampled sparse depth map.
            - down_valid_mask (torch.Tensor): The validity mask for the downsampled map.
    """
    if scale_factor == 1:
        return sparse_depth, (sparse_depth > 0.0)

    # Use average pooling for downsampling, it's more robust for sparse data
    # We treat invalid depths as 0. The denominator will average over valid points.
    kernel = torch.ones(1, 1, scale_factor, scale_factor, device=sparse_depth.device)

    # Sum of depths in a window
    sum_pooled = F.conv2d(sparse_depth, kernel, stride=scale_factor)

    # Count of valid depths in a window
    valid_mask = (sparse_depth > 0).float()
    count_pooled = F.conv2d(valid_mask, kernel, stride=scale_factor)

    # Average over valid depths
    avg_pooled_depth = sum_pooled / (count_pooled + 1e-8)

    # A downsampled pixel is valid if it contains at least one valid original pixel
    down_valid_mask = count_pooled > 0

    return avg_pooled_depth * down_valid_mask.float(), down_valid_mask

def compute_grad(x: torch.Tensor, thres: float = 3.0):
    """
    Computes the 2D gradient (dx, dy) of a batch of images and clips it.

    Args:
        x (torch.Tensor): The input image tensor. Shape: [batch_size, 1, h, w].
        thres (float): The threshold at which to clip the absolute gradient values.

    Returns:
        torch.Tensor: The gradient tensor with separate channels for dx and dy.
                      Shape: [batch_size, 2, h, w].
    """
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    grad = torch.stack([dx, dy], dim=1).squeeze(2)
    return torch.clamp(grad, min=-thres, max=thres)

def poisson_completion(
    sparse: torch.Tensor,
    mono_depth: torch.Tensor,
    num_scales: int = 4, # Number of pyramid levels
    thres: float = 3.0,
    lamda: float = 5.0,
    rtol: float = 1e-5,
    max_iter_per_scale: list[int] | None = None,
    confidence: torch.Tensor | None = None,
    max_resolution_ratio: float = 1.0,
):
    """
    Multi-scale Poisson-based depth completion.

    Args:
        sparse: [B, 1, H, W] - sparse depth (depth space, invalid=0).
        mono_depth: [B, 1, H, W] - monocular prediction.
        num_scales: Number of pyramid levels to use.
        thres: Gradient clipping threshold.
        lamda: Regularization weight.
        rtol: Relative tolerance for CG solver.
        max_iter_per_scale: List of max iterations for each scale, from coarsest to finest.
        max_resolution_ratio (float): The maximum resolution for the Poisson solver, specified as
                                      a ratio of the original height. For example, 1.0 means full
                                      resolution, 0.5 means half resolution. Defaults to 1.0.

    Returns:
        completed_depth: [B, 1, H, W] - completed depth map.
    """
    original_h, original_w = sparse.shape[-2:]
    # 1. Align monocular depth to sparse depth once at the beginning
    align_depth = least_square_align_lstsq_vectorized(sparse, mono_depth)

    # 2. Prepare iteration counts for each scale
    if max_iter_per_scale is None:
        # More iterations for coarser levels, fewer for finer levels
        max_iter_per_scale = [5000, 2000, 1000, 500]
        max_iter_per_scale = max_iter_per_scale[:num_scales]
    assert len(max_iter_per_scale) == num_scales, "Length of max_iter_per_scale must match num_scales."

    # 3. Determine the absolute max resolution from the ratio
    # Clamp ratio to be at most 1.0, as upsampling before solving is not intended.
    max_res_ratio = min(max_resolution_ratio, 1.0)
    # Calculate the absolute pixel value for the maximum resolution
    max_resolution_abs = int(original_h * max_res_ratio)

    # 4. Determine the finest scale to run the solver on based on the absolute resolution
    start_scale_idx = num_scales - 1
    stop_scale_idx = 0 # Default to finest scale (s=0)

    # Only limit the scale if the ratio is less than 1.0
    if max_res_ratio < 1.0:
        scale_resolutions = [original_h // (2**s) for s in range(num_scales)]

        # Find the index 's' of the first scale whose resolution is <= our target
        for s in range(num_scales):
            if scale_resolutions[s] <= max_resolution_abs:
                stop_scale_idx = s
                break
        else:
            # If the target resolution is smaller than the coarsest scale,
            # just run on the coarsest scale.
            stop_scale_idx = start_scale_idx

    # 5. Multi-scale iteration
    current_log_depth = None

    for s in range(start_scale_idx, stop_scale_idx - 1, -1):
        scale_factor = 2 ** s

        h_curr = original_h // scale_factor
        w_curr = original_w // scale_factor

        # a) Downsample sparse depth
        low_sparse, low_valid = downsample_sparse_depth(sparse, scale_factor=scale_factor)

        # b) Downsample aligned mono depth and compute gradient
        align_depth_scaled = F.interpolate(
            align_depth, size=(h_curr, w_curr), mode='bilinear', align_corners=False
        )
        gradient = compute_grad(log(align_depth_scaled), thres=thres)

        # c) Downsample confidence map
        confidence_scaled = None
        if confidence is not None:
            confidence_scaled = F.interpolate(
                confidence, size=(h_curr, w_curr), mode='bilinear', align_corners=False
            )

        # d) Prepare initial guess
        init_depth = None
        if current_log_depth is not None:
            init_depth = F.interpolate(
                current_log_depth, size=(h_curr, w_curr), mode='bilinear', align_corners=False
            )

        # --- Solve Poisson equation ---
        iter_idx = num_scales - 1 - s
        current_log_depth = poisson_solver(
            gradient=gradient,
            sparse_depth=log(low_sparse),
            valid_mask=low_valid,
            init_depth=init_depth,
            lamda=lamda,
            rtol=rtol,
            max_iter=max_iter_per_scale[iter_idx],
            gradient_confidence=confidence_scaled
        )

    # 6. Convert to depth space and upscale if necessary
    final_depth = current_log_depth.clamp(-10, 10).exp()

    if final_depth.shape[-2:] != (original_h, original_w):
        final_depth = F.interpolate(
            final_depth, size=(original_h, original_w), mode='bilinear', align_corners=False
        )

    final_depth = torch.nan_to_num(final_depth)
    return final_depth
