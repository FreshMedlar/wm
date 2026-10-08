"""Euler & Heun numerical ODE integrators for Flow Matching Bridge rollout.

Integrates from prior frame z_{t-1} at s=0 to predicted frame z_t at s=1:
    dz/ds = v_theta(z_s, s | z_ctx, z_guide, jepa_tokens)
"""

import torch
import torch.nn as nn


@torch.no_grad()
def solve_flow_bridge(
    model: nn.Module,
    z_init: torch.Tensor,
    z_ctx: torch.Tensor,
    z_guide: torch.Tensor,
    jepa_tokens: torch.Tensor,
    steps: int = 2,
    method: str = "heun",
) -> torch.Tensor:
    """Solves the Flow Matching ODE from s=0 (prior frame z_{t-1}) to s=1 (next frame z_t).

    Args:
        model: LatentEvolutionDiT model
        z_init: [B, 4, H, W] initial latent z_{t-1}
        z_ctx: [B, 8, H, W] context latents [z_{t-1}, z_{t-2}]
        z_guide: [B, 4, H, W] high-resolution guidance latent from JEPAUpsampler
        jepa_tokens: [B, N, D] LeV-JEPA semantic guidance tokens
        steps: Number of integration steps (typically 1 to 4)
        method: 'euler' or 'heun' (2nd-order Runge-Kutta)

    Returns:
        z_next: [B, 4, H, W] evolved latent at s=1
    """
    device = z_init.device
    batch_size = z_init.shape[0]

    z = z_init.clone()
    dt = 1.0 / steps

    for i in range(steps):
        s_val = i * dt
        s_tensor = torch.full((batch_size,), s_val, device=device, dtype=torch.float32)

        # 1. First vector field evaluation
        v1 = model(z, s_tensor, z_ctx, z_guide, jepa_tokens)

        if method == "euler" or steps == 1:
            z = z + v1 * dt
        elif method == "heun":
            # Predictor step
            z_pred = z + v1 * dt
            s_next = (i + 1) * dt
            s_next_tensor = torch.full((batch_size,), s_next, device=device, dtype=torch.float32)

            # Corrector step
            v2 = model(z_pred, s_next_tensor, z_ctx, z_guide, jepa_tokens)
            z = z + 0.5 * (v1 + v2) * dt
        else:
            raise ValueError(f"Unknown integration method: {method}")

    return z
