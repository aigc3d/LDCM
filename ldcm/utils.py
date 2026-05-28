import torch

def median(x: torch.Tensor, default_value: float = 1.0) -> torch.Tensor:
    """
    Compute the median of positive values for each sample in the batch.

    Args:
        x: Input tensor of shape [B, C, H, W] (or any ndim >= 2).
           Values <= 0 are ignored.
        default_value: Fallback value if no positive elements exist.

    Returns:
        Tensor of shape [B, 1, 1, ..., 1] with median (or default) per batch.
    """
    x_masked = torch.where(x > 0.0, x, torch.nan)

    x_flat = x_masked.flatten(start_dim=1)  # Shape: [B, C*H*W]
    medians = torch.nanmedian(x_flat, dim=1).values  # Shape: [B]
    medians = torch.where(torch.isnan(medians),
                          torch.tensor(default_value, device=x.device, dtype=x.dtype),
                          medians)
    ndim = x.ndim
    return medians.view(-1, *(1,) * (ndim - 1))

def log(x: torch.Tensor) -> torch.Tensor:
    """
    Safe natural logarithm: computes log(x) for x > 0, returns 0 otherwise.
    """
    return torch.where(x > 0, torch.log(x), torch.zeros_like(x))
