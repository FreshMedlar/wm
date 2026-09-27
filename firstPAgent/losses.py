"""
Mathematical Loss Functions for First-Principles Active Inference Agent.

Includes:
- SIGReg (Anti-Collapse Regularizer): Variance & Covariance de-correlation loss.
- Latent Space Health Monitoring metrics.
"""

from typing import Dict
import torch
import torch.nn.functional as F


def sigreg_loss(
    z: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.05,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Computes the SIGReg anti-collapse regularizer across a batch of active representations Z.

    Formulation:
      mu_j = 1/B * sum_{i=1}^B Z_{i, j}
      Sigma = 1/(B - 1) * sum_{i=1}^B (Z_i - mu)(Z_i - mu)^T
      L_std = 1/D * sum_{j=1}^D max(0, 1 - sqrt(Sigma_{j, j} + eps))
      L_cov = 1/D * sum_{i != j} Sigma_{i, j}^2
      L_SIGReg = alpha * L_std + beta * L_cov

    Args:
        z: Active latent representations tensor of shape (B, D).
        alpha: Weight for standard deviation regularizer (prevents variance collapse). Default: 1.0.
        beta: Weight for covariance regularizer (decorrelates latent dimensions). Default: 0.05.
        eps: Small epsilon for numerical stability when calculating square root. Default: 1e-4.

    Returns:
        Scalar torch.Tensor containing the weighted SIGReg loss.
    """
    B, D = z.shape
    if B <= 1:
        # Cannot calculate sample covariance with batch size <= 1
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # 1. Centered latents
    z_centered = z - z.mean(dim=0, keepdim=True)

    # 2. Sample covariance matrix: (D, D)
    cov = (z_centered.T @ z_centered) / (B - 1)

    # 3. Variance loss (L_std)
    std = torch.sqrt(torch.diagonal(cov) + eps)
    loss_std = torch.mean(F.relu(1.0 - std))

    # 4. Off-diagonal covariance loss (L_cov)
    # Sum of all squared elements minus sum of diagonal squared elements
    loss_cov = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / D

    # 5. Combined SIGReg loss
    return alpha * loss_std + beta * loss_cov


def compute_latent_metrics(z: torch.Tensor, eps: float = 1e-4) -> Dict[str, float]:
    """
    Computes diagnostic metrics for latent space health monitoring:
    - mean std across dimensions (target: ~1.0)
    - mean off-diagonal covariance (target: < 0.1)
    - max off-diagonal covariance

    Args:
        z: Latent tensor of shape (B, D).
        eps: Epsilon for std calculation.

    Returns:
        Dict with 'std_mean', 'off_diag_mean', 'off_diag_max'.
    """
    with torch.no_grad():
        B, D = z.shape
        if B <= 1:
            return {"std_mean": 0.0, "off_diag_mean": 0.0, "off_diag_max": 0.0}

        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / (B - 1)

        std = torch.sqrt(torch.diagonal(cov) + eps)
        std_mean = std.mean().item()

        # Mask diagonal
        diag_mask = torch.eye(D, dtype=torch.bool, device=z.device)
        off_diag = cov[~diag_mask].abs()

        off_diag_mean = off_diag.mean().item() if off_diag.numel() > 0 else 0.0
        off_diag_max = off_diag.max().item() if off_diag.numel() > 0 else 0.0

        return {
            "std_mean": std_mean,
            "off_diag_mean": off_diag_mean,
            "off_diag_max": off_diag_max,
        }
