"""Single-scale weighted Poisson solver with cached CUDA graph CG.

The accelerated and PyTorch paths retain the tested arithmetic and batch-wide
stopping rules. CUDA kernels are imported only when a CUDA FP32 solve needs
one, so importing this module and solving on CPU do not require Triton.

Derived from ViGeo's Apache-2.0 finite-difference/CG implementation and its
accelerated solver. Modified to be self-contained in this package; see LICENSE.
Integrated into LDCM with an optional Triton backend.
"""
from collections import OrderedDict
from functools import lru_cache
import threading

import torch

from ._native import backend_info


@lru_cache(maxsize=1)
def _get_kernels():
    """Load optional Triton kernels once; unrelated import errors stay visible."""
    try:
        from . import _kernels
    except ModuleNotFoundError as error:
        if error.name == "triton" or (error.name or "").startswith("triton."):
            return None
        raise
    return _kernels

class _FiniteDifference:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def bmm(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, self.h, self.w)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gradients = torch.cat([dx.reshape(batch_size, -1), dy.reshape(batch_size, -1)], dim=1)
        return gradients.unsqueeze(-1)

    def bmm_transposed(self, x: torch.Tensor):
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


def _flatten_gradient(batched_matrix: torch.Tensor):
    batch_size, _, h, w = batched_matrix.shape
    x_truncated = batched_matrix[:, 0, :, :-1].reshape(batch_size, -1)
    y_truncated = batched_matrix[:, 1, :-1, :].reshape(batch_size, -1)
    return torch.cat([x_truncated, y_truncated], dim=-1).unsqueeze(-1)


def _cg(A_bmm, B, X0, rtol, maxiter):
    """Unpreconditioned CG with the completion algorithm's fixed stop rules."""
    X_k = X0
    R_k = B - A_bmm(X_k)
    P_k = R_k

    B_norm = torch.norm(B, dim=1, keepdim=True)
    stopping_matrix = torch.max(rtol * B_norm, torch.zeros_like(B_norm))

    residual_norm = torch.norm(R_k, dim=1, keepdim=True)
    residual_norm_last = residual_norm.clone()

    consecutive_no_improvement = 0

    for k in range(1, maxiter + 1):
        Ap = A_bmm(P_k)
        inner_RZ = torch.sum(R_k * R_k, dim=1, keepdim=True)
        denominator = torch.sum(P_k * Ap, dim=1, keepdim=True)
        denominator = torch.where(denominator == 0, torch.tensor(1e-8, device=B.device), denominator)
        alpha = inner_RZ / denominator

        X_k = X_k + alpha * P_k
        R_k_new = R_k - alpha * Ap

        residual_norm = torch.norm(R_k_new, dim=1, keepdim=True)

        if (residual_norm <= stopping_matrix).all():
            break

        rel_improvement = (residual_norm_last - residual_norm) / (residual_norm_last + 1e-8)
        if rel_improvement.max() < 1e-4:
            consecutive_no_improvement += 1
        else:
            consecutive_no_improvement = 0

        if consecutive_no_improvement >= 10:
            break

        residual_norm_last = residual_norm.clone()

        beta_denom = torch.where(inner_RZ == 0, torch.tensor(1e-8, device=B.device), inner_RZ)
        beta = torch.sum(R_k_new * R_k_new, dim=1, keepdim=True) / beta_denom
        P_k = R_k_new + beta * P_k

        R_k = R_k_new

    return X_k


def _poisson_solver_torch(gradient, sparse_depth, valid_mask, gradient_confidence,
                          sparse_confidence, init_depth, lamda, rtol, max_iter):
    batch_size, _, h, w = gradient.shape
    A = _FiniteDifference(h, w)

    gradient_flattened = _flatten_gradient(gradient)
    gradient_confidence_flattened = _flatten_gradient(gradient_confidence)
    b_bottom = A.bmm_transposed(gradient_confidence_flattened * gradient_flattened)
    b_top = lamda * (sparse_confidence * valid_mask * sparse_depth).reshape(batch_size, -1, 1)
    b = b_top + b_bottom

    def laplacian_and_mass_matrix_bmm(p):
        grad_term = A.bmm_transposed(gradient_confidence_flattened * A.bmm(p))
        ap = sparse_confidence * valid_mask * p.reshape(batch_size, 1, h, w)
        sparse_term = lamda * ap.reshape(batch_size, -1, 1)
        return grad_term + sparse_term

    init_depth_flat = (init_depth if init_depth is not None else sparse_depth * valid_mask).reshape(batch_size, -1, 1)
    x = _cg(laplacian_and_mass_matrix_bmm, b, X0=init_depth_flat, rtol=rtol, maxiter=max_iter)
    return x.reshape(sparse_depth.shape)


_local = threading.local()
_construction_lock = threading.Lock()
_CACHE_LIMIT = 16
_BLOCK = 8


class _GraphCG:
    def __init__(self, batch, height, width, device, block=None):
        self.shape = (batch, height * width, 1)
        self.height, self.width = height, width
        from ._native import get_backend
        self.native = get_backend()
        self.kernels = _get_kernels()
        if block is None:
            block = _BLOCK if self.native is not None else 16
        self.block = block
        self.x = torch.zeros(self.shape, device=device, dtype=torch.float32)
        self.r = torch.zeros_like(self.x)
        self.p = torch.zeros_like(self.x)
        self.rhs = torch.zeros_like(self.x)
        self.rtol = torch.full((),1e-5,device=device,dtype=torch.float32)
        self.mass = torch.ones_like(self.x)
        self.weight = torch.ones((batch, height*(width-1)+(height-1)*width, 1), device=device, dtype=torch.float32)
        self.lamda = torch.full((batch, 1, 1), 5., device=device, dtype=torch.float32)
        self.threshold = torch.zeros((batch, 1, 1), device=device, dtype=torch.float32)
        self.last_norm = torch.zeros_like(self.threshold)
        self.active = torch.ones((), device=device, dtype=torch.bool)
        self.steps = torch.zeros((), device=device, dtype=torch.int32)
        self.stalled = torch.zeros_like(self.steps)
        self.cap = torch.full_like(self.steps, 5000)
        self.rho = torch.zeros((batch, 1, 1), device=device, dtype=torch.float32)
        self.beta = torch.zeros_like(self.rho)
        self.graph = torch.cuda.CUDAGraph()
        self.initialization_graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.initialize()
                self.step()
        current.wait_stream(stream)
        with torch.cuda.graph(self.initialization_graph,stream=stream,capture_error_mode='thread_local'):
            self.initialize()
        with torch.cuda.graph(self.graph, stream=stream, capture_error_mode='thread_local'):
            for _ in range(block):
                self.step()
        current.wait_stream(stream)

    def matvec(self, value):
        return self.kernels.matvec(value, self.weight, self.mass, self.lamda, self.height, self.width)

    def step(self):
        kernels = self.kernels
        ap,products = kernels.matvec_products(self.p,self.weight,self.mass,self.lamda,self.height,self.width)
        denominator = torch.sum(products, dim=1, keepdim=True)
        kernels.update_x_r(self.x,self.r,self.p,ap,self.rho,denominator,self.active)
        if self.native is not None and self.shape[0] == 1:
            self.native.residual_control(self.r,self.last_norm,self.threshold,self.rho,
                self.beta,self.active,self.steps,self.stalled,self.cap)
        else:
            if self.native is not None:
                new_rho,norm = self.native.residual_stats(self.r)
            else:
                norm = torch.norm(self.r, dim=1, keepdim=True)
                new_rho = torch.sum(self.r * self.r, dim=1, keepdim=True)
            kernels.control(norm,self.last_norm,self.threshold,self.rho,new_rho,self.beta,
                    self.active,self.steps,self.stalled,self.cap)
        kernels.update_p(self.p,self.r,self.beta,self.active)

    def solve(self, rhs, initial, weight, mass, lamda, rtol, max_iter):
        self.weight.copy_(weight)
        self.mass.copy_(mass)
        self.lamda.copy_(lamda)
        self.x.copy_(initial)
        self.rhs.copy_(rhs)
        self.rtol.fill_(rtol)
        self.cap.fill_(max_iter)
        self.initialization_graph.replay()
        while True:
            self.graph.replay()
            if not self.active.item():
                break
        return self.x.clone()

    def initialize(self):
        self.r.copy_(self.rhs - self.matvec(self.x))
        self.p.copy_(self.r)
        if self.native is not None:
            rho,norm = self.native.residual_stats(self.r)
        else:
            rho = torch.sum(self.r * self.r, dim=1, keepdim=True)
            norm = torch.norm(self.r, dim=1, keepdim=True)
        self.rho.copy_(rho)
        self.last_norm.copy_(norm)
        rhs_norm = torch.norm(self.rhs, dim=1, keepdim=True)
        self.threshold.copy_(torch.max(self.rtol*rhs_norm, torch.zeros_like(rhs_norm)))
        self.active.fill_(True)
        self.steps.zero_()
        self.stalled.zero_()


def _cache():
    if not hasattr(_local, 'cache'):
        _local.cache = OrderedDict()
    return _local.cache


def clear_cache():
    """Release the current thread's cached CUDA graphs and workspaces."""
    with _construction_lock:
        _cache().clear()


def cache_info():
    """Return shapes and most recent iteration counts for diagnostics."""
    return [dict(key=list(key), iterations=int(value.steps.item())) for key,value in _cache().items()]


def _workspace(batch, height, width, device):
    cache = _cache()
    key = (device.index, torch.cuda.current_stream(device).cuda_stream, batch, height, width)
    if key not in cache:
        # torch.cuda.graph synchronizes the device on entry. Another thread
        # must not enter that context while one of our graphs is capturing.
        # Cached solves do not acquire this lock.
        # Validation may run first under inference_mode. Persistent buffers
        # must remain writable when subsequent training reuses this workspace.
        with (_construction_lock, torch.cuda.device(device),
              torch.inference_mode(False), torch.no_grad()):
            cache[key] = _GraphCG(batch, height, width, device)
            while len(cache) > _CACHE_LIMIT:
                cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


@torch.no_grad()
def poisson_solver(gradient, sparse_depth, valid_mask, gradient_confidence,
                   sparse_confidence, init_depth, lamda, rtol, max_iter):
    """Internal solve for the tensors and scalar parameters prepared by core.

    CUDA FP32 with Triton uses cached CUDA graphs; otherwise use PyTorch.
    Confidence maps already have the required channels and positive weights.
    """
    batch, _, height, width = gradient.shape
    lamda = torch.as_tensor(lamda, device=gradient.device, dtype=gradient.dtype).expand(batch, 1, 1)
    if not gradient.is_cuda or gradient.dtype != torch.float32 or _get_kernels() is None:
        return _poisson_solver_torch(gradient, sparse_depth, valid_mask, gradient_confidence,
                                  sparse_confidence, init_depth, lamda, rtol, max_iter)
    weight = _flatten_gradient(gradient_confidence)
    grad = _flatten_gradient(gradient)
    operator = _FiniteDifference(height, width)
    bottom = operator.bmm_transposed(weight * grad)
    mass = (sparse_confidence * valid_mask).reshape(batch, -1, 1)
    top = lamda * (mass * sparse_depth.reshape(batch, -1, 1))
    rhs = top + bottom
    initial = (init_depth if init_depth is not None else sparse_depth * valid_mask).reshape(batch, -1, 1)
    state = _workspace(batch, height, width, gradient.device)
    result = state.solve(rhs, initial, weight, mass, lamda, rtol, max_iter)
    return result.reshape(sparse_depth.shape)
