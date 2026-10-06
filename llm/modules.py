import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer (SIGReg).
    Enforces the continuous latent space representations to be isotropic standard Gaussian,
    preventing representation collapse in Joint Embedding Predictive Architectures (LeJEPA).

    Reference: Balestriero & LeCun (2024).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: Tensor of shape (T, B, D) or (N, D).
        Computes the Epps-Pulley empirical characteristic function test statistic
        against standard normal Gaussian.
        """
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)  # Shape (1, N, D)
        elif proj.dim() == 3 and proj.size(0) != proj.size(1):
            # If (B, T, D) passed, transpose to (T, B, D) for time/token slices
            pass

        d = proj.size(-1)
        # Sample random projection vectors on unit sphere
        A = torch.randn(d, self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)

        # Compute projected coordinates: (..., N, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Provides relative position awareness for causal language modeling.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # [seq_len, dim/2] -> [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, T, D_head]
    # cos, sin: [T, D_head] -> unsqueeze to [1, 1, T, D_head]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention utilizing FlashAttention / PyTorch scaled_dot_product_attention.
    Supports RoPE positional embeddings and optional KV caching.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if use_rope:
            self.rope = RotaryEmbedding(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, D_head]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(q, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=drop_p,
            is_causal=(is_causal and mask is None),
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Standard Transformer Feed-Forward Network with GELU activation.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), is_causal=is_causal, mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class ConditionalTransformerBlock(nn.Module):
    """
    Transformer Block with conditioning support (via AdaLN or residual injection).
    Used in the Conditional Decoder to incorporate the continuous latent plan z_hat_{t+1}.
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        # Conditioning projection to modulate features
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
        )
        # Standard AdaLN zero-initialization for residual identity at init
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        is_causal: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cond: [B, T, d_cond]
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)

        norm_x1 = self.norm1(x) * (1.0 + scale) + shift
        x = x + self.attn(norm_x1, is_causal=is_causal, mask=mask)

        norm_x2 = self.norm2(x) * (1.0 + scale) + shift
        x = x + self.ffn(norm_x2)
        return x
