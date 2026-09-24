import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, is_causal=False):
        B, seq_len, _ = q.shape

        # Linear projections & split into heads: [B, seq_len, n_heads, d_k] -> [B, n_heads, seq_len, d_k]
        q = self.q_proj(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Uses PyTorch's native Scaled Dot-Product Attention (FlashAttention under the hood on GPUs)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            is_causal=is_causal
        )

        # Concatenate heads and project out: [B, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, is_causal=False):
        # Attention block with residual connection (Pre-LayerNorm)
        norm_x = self.norm1(x)
        x = x + self.dropout(self.attn(norm_x, norm_x, norm_x, is_causal=is_causal))
        
        # Feed-forward block with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class LeJEPAEncoder(nn.Module):
    """
    Streaming/Causal Encoder that maps token sequences 
    into continuous latent states z_t without vocabulary logits.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        d_latent: int = 256, 
        n_heads: int = 8, 
        d_ff: int = 2048, 
        num_layers: int = 6, 
        max_len: int = 2048
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # LeJEPA projection head: maps hidden features to the isotropic latent space
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:  [Batch, Seq_Len]
        Output: [Batch, Seq_Len, d_latent] -> Sequence of continuous states (z_1, ..., z_t)
        """
        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        
        # MUST BE CAUSAL: token t cannot see future tokens
        for layer in self.layers:
            x = layer(x, is_causal=True)
            
        x = self.norm(x)
        z = self.proj_head(x)  # [Batch, Seq_Len, d_latent]
        return z


class LatentPredictor(nn.Module):
    """
    World Model Dynamics Predictor P_phi:
    Takes current latent z_t and a horizon step k, predicts z_{t+k} in continuous space.
    """
    def __init__(self, d_latent: int = 256, d_hidden: int = 512, num_layers: int = 4):
        super().__init__()
        # Step/horizon conditioning embedding (scalar k -> dense embedding)
        self.step_embed = nn.Sequential(
            nn.Linear(1, d_latent),
            nn.SiLU(),
            nn.Linear(d_latent, d_latent)
        )
        
        layers = []
        in_dim = d_latent
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, d_hidden),
                nn.LayerNorm(d_hidden),
                nn.GELU(),
            ])
            in_dim = d_hidden
        layers.append(nn.Linear(d_hidden, d_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        z_t: [Batch, d_latent]
        k:   [Batch, 1] (horizon steps into the future)
        """
        # Condition the latent state on the prediction horizon
        cond = z_t + self.step_embed(k)
        z_hat_next = self.net(cond)
        return z_hat_next


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device = "cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
