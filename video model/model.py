"""Latent Flow-Matching Evolution Network Guided by High-Resolution JEPA Upsampler.

Implements:
1. SpatialGuideProjector: Projects high-resolution 32x32 latent guide (from JEPAUpsampler)
   into the spatial tokens for the DiT.
2. LatentEvolutionDiT: Flow-Matching Velocity Predictor v_theta(z_s, s | [z_{t-1}, z_{t-2}], z_guide, jepa_target)
   Predicts the straight vector field u_t = z_t - z_{t-1}.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """Modulates tensor x with scale and shift parameters (AdaLN-Zero)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar flow integration time s in [0, 1] into a high-dimensional vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class SpatialGuideProjector(nn.Module):
    """Projects high-resolution 32x32 guidance latent into cross-attention tokens with 2D coords."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 256):
        super().__init__()
        # 4 latent channels + 2 coordinate grid channels = 6
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + 2, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(),
        )

    def forward(self, z_guide: torch.Tensor) -> torch.Tensor:
        """Args:

            z_guide: [B, 4, H, W]
        Returns:
            guide_tokens: [B, H*W, embed_dim]
        """
        B, C, H, W = z_guide.shape
        ys = torch.linspace(-1.0, 1.0, steps=H, device=z_guide.device)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=z_guide.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        feat = self.proj(torch.cat([z_guide, coords], dim=1))  # [B, embed_dim, H, W]
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        return tokens


class DiTBlock(nn.Module):
    """Diffusion/Flow Transformer block with AdaLN-Zero and Spatial Guide Cross-Attention."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN modulation projections (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Input latent tokens [B, N, D]
            c: Conditioning vector (time embedding + pooled JEPA) [B, D]
            context: Aligned high-res guidance tokens [B, N, D]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # 1. Self-Attention with AdaLN modulation
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        sa_out, _ = self.attn(normed, normed, normed)
        x = x + gate_msa.unsqueeze(1) * sa_out

        # 2. Cross-Attention with High-Res Guide
        normed_cross = self.norm_cross(x)
        ca_out, _ = self.cross_attn(normed_cross, context, context)
        x = x + ca_out

        # 3. Feed-Forward MLP with AdaLN modulation
        normed_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class LatentEvolutionDiT(nn.Module):
    """Flow-Matching Evolution DiT Network Guided by Upsampled JEPA Latents.

    Predicts velocity field v_theta along the bridge path between z_{t-1} and z_t:
        u_t = z_t - z_{t-1}

    Inputs:
        z_s: Current bridge state [B, 4, H_lat, W_lat] at flow time s in [0, 1]
        s: Flow time step [B]
        z_ctx: Concatenated past context latents [z_{t-1}, z_{t-2}] [B, 8, H_lat, W_lat]
        z_guide: High-resolution guidance latent from JEPAUpsampler [B, 4, H_lat, W_lat]
        jepa_tokens: Raw patch tokens from frozen LeV-JEPA [B, 196 or 197, 1024]
    """

    def __init__(
        self,
        latent_channels: int = 4,
        context_frames: int = 2,
        latent_size: int = 32,
        hidden_size: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        jepa_dim: int = 1024,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.context_frames = context_frames
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        # In channels = noisy bridge state (4) + context latents (8) + high-res guide (4) = 16 channels!
        in_channels = latent_channels * (1 + context_frames + 1)

        # Patch projection (1x1 conv to project spatial grid into DiT hidden dimension)
        self.x_proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, latent_size * latent_size, hidden_size))

        # Time step embedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Global JEPA pooler for AdaLN conditioning
        self.jepa_global_proj = nn.Linear(jepa_dim, hidden_size)

        # Spatial Guide Projector for cross-attention
        self.guide_proj = SpatialGuideProjector(in_channels=latent_channels, embed_dim=hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # Final projection layer (AdaLN-Zero)
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.final_linear = nn.Linear(hidden_size, latent_channels, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(
        self,
        z_s: torch.Tensor,
        s: torch.Tensor,
        z_ctx: torch.Tensor,
        z_guide: torch.Tensor,
        jepa_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

            z_s: [B, 4, H, W] bridge latent
            s: [B] flow time in [0, 1]
            z_ctx: [B, 8, H, W] context latents [z_{t-1}, z_{t-2}]
            z_guide: [B, 4, H, W] high-resolution guidance from upsampler
            jepa_tokens: [B, 196 or 197, 1024] LeV-JEPA features
        Returns:
            v_pred: [B, 4, H, W] predicted flow velocity
        """
        B, _, H, W = z_s.shape

        # 1. Combine bridge state with temporal history AND high-res guide along channels
        x_in = torch.cat([z_s, z_ctx, z_guide], dim=1)  # [B, 16, H, W]
        x = self.x_proj(x_in)                           # [B, hidden_size, H, W]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.hidden_size)
        x = x + self.pos_embed[:, : H * W, :]

        # 2. Conditionings: Time embedding + Global JEPA pooled embedding
        t_emb = self.t_embedder(s)                      # [B, hidden_size]
        if jepa_tokens.shape[1] == 197:
            jepa_tokens_pool = jepa_tokens[:, 1:, :]
        else:
            jepa_tokens_pool = jepa_tokens
        jepa_pooled = jepa_tokens_pool.mean(dim=1)      # [B, jepa_dim]
        c = t_emb + self.jepa_global_proj(jepa_pooled)

        # 3. High-resolution guidance cross-attention tokens
        context_tokens = self.guide_proj(z_guide)       # [B, H*W, hidden_size]

        # 4. DiT blocks forward
        for block in self.blocks:
            x = block(x, c, context_tokens)

        # 5. Output projection
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        out = self.final_linear(x)                      # [B, H*W, 4]

        # Reshape back to 2D latent map [B, 4, H, W]
        out = out.reshape(B, H, W, self.latent_channels).permute(0, 3, 1, 2)
        return out
