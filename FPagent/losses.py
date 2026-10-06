"""
Mathematical Loss Functions for First-Principles Active Inference Agent.

Includes:
- SIGReg (Sketch Isotropic Gaussian Regularizer): Epps-Pulley characteristic function
  test for distribution matching to isotropic standard normal N(0, I).
- Latent Space Health Monitoring metrics.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (SIGReg) as formulated in LeWorldModel / LeJEPA.
    Measures the Epps-Pulley test statistic between the empirical distribution
    of latent projections and a standard 1D Gaussian N(0, 1) using numerical quadrature knots.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Computes the Epps-Pulley statistic.

        Args:
            proj: Latent tensor of shape (B, D) or (T, B, D).

        Returns:
            Scalar loss representing distance to standard isotropic Gaussian.
        """
        if proj.ndim == 2:
            # (B, D) -> (1, B, D) to unify with (T, B, D)
            proj = proj.unsqueeze(0)

        # Sample random 1D projections on the unit sphere
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True))

        # Project and evaluate empirical characteristic function against N(0, 1)
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def sigreg_loss(
    z: torch.Tensor,
    knots: int = 17,
    num_proj: int = 1024,
) -> torch.Tensor:
    """
    Functional interface for true SIGReg Epps-Pulley isotropic Gaussian regularization.
    """
    reg = SIGReg(knots=knots, num_proj=num_proj).to(z.device)
    return reg(z)


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
