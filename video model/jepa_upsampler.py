"""High-Resolution Guided JEPA Latent Upsampler.

Translates:
    Inputs:
        z_{t-1}: Previous sharp VAE latent [B, 4, 32, 32]
        z_t^{jepa}: Target frame LeV-JEPA tokens [B, 196, 1024]
    Output:
        ẑ_t: High-resolution predicted VAE latent [B, 4, 32, 32]

Uses multi-scale residual blocks, progressive spatial upsampling (14x14 -> 32x32),
explicit 2D coordinate meshgrids, and skip connections from z_{t-1} to preserve crisp edges.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual convolutional block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + res)


class JEPAUpsampler(nn.Module):
    """Guided Latent Upsampler from LeV-JEPA tokens and previous frame latent."""

    def __init__(
        self,
        latent_channels: int = 4,
        jepa_dim: int = 1024,
        hidden_dim: int = 128,
        target_size: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.target_size = target_size

        # 1. High-frequency encoder for previous frame latent z_{t-1} (32x32)
        # Input channels: 4 (latent) + 2 (coordinate grid) = 6
        self.prev_encoder = nn.Sequential(
            nn.Conv2d(latent_channels + 2, hidden_dim, kernel_size=3, padding=1),
            ResBlock(hidden_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
        )

        # 2. Spatial projector & upsampler for LeV-JEPA tokens (14x14 -> 32x32)
        self.jepa_proj = nn.Sequential(
            nn.Conv2d(jepa_dim, hidden_dim * 2, kernel_size=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            # 28x28 -> 32x32
            nn.Upsample(size=(target_size, target_size), mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        # 3. Fusion & Multi-Scale Decoder (guided advection & inpainting)
        # Combines: prev_features (hidden_dim) + jepa_features (hidden_dim) = hidden_dim * 2
        self.fusion = nn.Sequential(
            ResBlock(hidden_dim * 2, hidden_dim * 2),
            ResBlock(hidden_dim * 2, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
        )

        # 4. Output head: predicts residual shift delta_z from z_{t-1}
        self.out_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, latent_channels, kernel_size=3, padding=1),
        )

    def _get_coord_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns normalized [-1, 1] 2D coordinate grid of shape [B, 2, H, W]."""
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        return coords

    def forward(self, z_prev: torch.Tensor, jepa_tokens: torch.Tensor) -> torch.Tensor:
        """Args:

            z_prev: [B, 4, 32, 32] previous frame VAE latent
            jepa_tokens: [B, 196 or 197, 1024] LeV-JEPA patch tokens
        Returns:
            z_pred: [B, 4, 32, 32] next frame high-res predicted latent
        """
        B, C, H, W = z_prev.shape

        # Drop CLS token if present
        if jepa_tokens.shape[1] == 197:
            jepa_tokens = jepa_tokens[:, 1:, :]

        # 1. Encode previous frame with coordinate grid
        coords = self._get_coord_grid(B, H, W, z_prev.device)
        z_in = torch.cat([z_prev, coords], dim=1)  # [B, 6, 32, 32]
        feat_prev = self.prev_encoder(z_in)        # [B, hidden_dim, 32, 32]

        # 2. Reshape and upsample LeV-JEPA tokens from 14x14 -> 32x32
        # jepa_tokens: [B, 196, 1024] -> [B, 1024, 14, 14]
        feat_jepa_2d = jepa_tokens.permute(0, 2, 1).reshape(B, 1024, 14, 14)
        feat_jepa_up = self.jepa_proj(feat_jepa_2d)  # [B, hidden_dim, 32, 32]

        # 3. Fuse previous textures with target guidance
        fused = torch.cat([feat_prev, feat_jepa_up], dim=1)  # [B, hidden_dim * 2, 32, 32]
        decoded = self.fusion(fused)                         # [B, hidden_dim, 32, 32]

        # 4. Predict residual: z_t = z_{t-1} + delta_z
        delta_z = self.out_head(decoded)                     # [B, 4, 32, 32]
        z_pred = z_prev + delta_z

        return z_pred
