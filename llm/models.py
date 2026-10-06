import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.modules import (
    CausalSelfAttention,
    TransformerBlock,
    ConditionalTransformerBlock,
    SIGReg,
)


class LeJEPAEncoder(nn.Module):
    """
    Causal Transformer Encoder mapping token sequences x_{\le t} into continuous
    latent representations z_t.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        d_latent: int = 128,
        n_heads: int = 8,
        num_layers: int = 7,
        dropout: float = 0.0,
        shared_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_latent = d_latent

        if shared_embedding is not None:
            self.tok_emb = shared_embedding
        else:
            self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Projection head to isotropic latent space governed by SIGReg
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: tokens of shape [B, T]
        Output:
            z: [B, T, d_latent] - continuous latent representations
            h: [B, T, d_model]  - hidden state before projection
        """
        x = self.tok_emb(tokens)
        for block in self.blocks:
            x = block(x, is_causal=True)
        h = self.norm(x)
        z = self.proj_head(h)
        return z, h


class LeJEPAPredictor(nn.Module):
    """
    Predictor module for LeJEPA.
    Given contextual representations of x_{\le t}, predicts the continuous
    latent representation \hat{z}_{t+1} of the next state.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_latent: int = 128,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent

        # Input projection from latent space or encoder hidden state
        self.in_proj = nn.Linear(d_latent, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_latent)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Input: z_seq of shape [B, T, d_latent] (latents up to time step t)
        Output: z_hat of shape [B, T, d_latent] (prediction of next latent z_{t+1})
        """
        x = self.in_proj(z_seq)
        for block in self.blocks:
            x = block(x, is_causal=True)
        x = self.norm(x)
        z_hat = self.out_proj(x)
        return z_hat


class ConditionalDecoder(nn.Module):
    """
    Autoregressive generator conditioned on both past discrete tokens x_{\le t}
    and the predicted continuous latent plan \hat{z}_{t+1}:

        P(x_{t+1} | x_{\le t}, \hat{z}_{t+1})

    Parameter budget: Intentionally lightweight (~10% of total parameters)
    so the LeJEPA planner carries the majority of model capacity (90-10 split).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        d_latent: int = 128,
        n_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
        shared_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_latent = d_latent

        if shared_embedding is not None:
            self.tok_emb = shared_embedding
        else:
            self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Project predicted latent plan \hat{z}_{t+1} into decoder hidden dimension
        self.latent_proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Causal transformer blocks with conditional modulation from \hat{z}_{t+1}
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(
                d_model=d_model,
                d_cond=d_latent,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Language Model vocabulary head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying with token embedding if shapes match
        if shared_embedding is not None:
            self.lm_head.weight = shared_embedding.weight

    def forward(
        self,
        tokens: torch.Tensor,
        z_hat_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        tokens:     [B, T] - Past discrete tokens x_{\le t}
        z_hat_next: [B, T, d_latent] - Predicted next continuous latent plan \hat{z}_{t+1}

        Returns logits of shape [B, T, vocab_size] for predicting x_{t+1}.
        """
        # Embed past tokens and fuse with the predicted plan vector
        tok_feat = self.tok_emb(tokens)
        latent_feat = self.latent_proj(z_hat_next)

        # Injected conditioning at input layer
        x = tok_feat + latent_feat

        # Pass through conditional transformer blocks
        for block in self.blocks:
            x = block(x, cond=z_hat_next, is_causal=True)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class LeJEPALanguageModel(nn.Module):
    """
    Two-Level Hierarchical Language Model:
      Level 1: LeJEPA Transformer (Causal Encoder + Predictor) acts as continuous prior/planner.
      Level 2: Conditional Decoder acts as autoregressive generator:
               P(x_{t+1} | x_{\le t}, \hat{z}_{t+1})

    Parameter distribution: ~90% parameters in LeJEPA, ~10% in Decoder.
    Training scheme: Scheme A Two-Stage (Strict JEPA) with L2 prediction loss + SIGReg,
    followed by decoder cross-entropy with stop-gradient on \hat{z}_{t+1}.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        d_latent: int = 128,
        n_heads: int = 8,
        lejepa_encoder_layers: int = 7,
        lejepa_predictor_layers: int = 2,
        decoder_layers: int = 1,
        dropout: float = 0.0,
        sigreg_weight: float = 0.1,
        share_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_latent = d_latent
        self.sigreg_weight = sigreg_weight

        # Shared token embedding table
        self.shared_emb = nn.Embedding(vocab_size, d_model) if share_embeddings else None

        # Level 1: LeJEPA Prior / Planner (~90% parameters)
        self.encoder = LeJEPAEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_latent=d_latent,
            n_heads=n_heads,
            num_layers=lejepa_encoder_layers,
            dropout=dropout,
            shared_embedding=self.shared_emb,
        )

        self.predictor = LeJEPAPredictor(
            d_model=d_model,
            d_latent=d_latent,
            n_heads=n_heads,
            num_layers=lejepa_predictor_layers,
            dropout=dropout,
        )

        # Regularizer to ensure isotropic Gaussian latent space and prevent collapse
        self.sigreg = SIGReg()

        # Level 2: Conditional Decoder (~10% parameters)
        self.decoder = ConditionalDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_latent=d_latent,
            n_heads=n_heads,
            num_layers=decoder_layers,
            dropout=dropout,
            shared_embedding=self.shared_emb,
        )

        # Standard transformer weight initialization (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """Calculates exact parameter counts and percentage breakdown."""
        jepa_params = sum(p.numel() for p in self.encoder.parameters()) + sum(
            p.numel() for p in self.predictor.parameters()
        )
        # Note: If embeddings are shared, avoid double-counting in decoder
        decoder_unique_params = sum(
            p.numel() for name, p in self.decoder.named_parameters()
            if not (self.shared_emb is not None and "tok_emb" in name)
        )
        total_unique = jepa_params + decoder_unique_params

        jepa_pct = (jepa_params / total_unique) * 100.0 if total_unique > 0 else 0.0
        dec_pct = (decoder_unique_params / total_unique) * 100.0 if total_unique > 0 else 0.0

        return {
            "jepa_params": jepa_params,
            "decoder_params": decoder_unique_params,
            "total_params": total_unique,
            "jepa_pct": jepa_pct,
            "decoder_pct": dec_pct,
        }

    def freeze_lejepa(self, freeze: bool = True):
        """Freezes Level 1 (LeJEPA Encoder blocks, norm, proj_head, and Predictor)."""
        # Freeze transformer representation backbone of LeJEPA
        for p in self.encoder.blocks.parameters():
            p.requires_grad = not freeze
        for p in self.encoder.norm.parameters():
            p.requires_grad = not freeze
        for p in self.encoder.proj_head.parameters():
            p.requires_grad = not freeze

        # Freeze continuous dynamics predictor
        for p in self.predictor.parameters():
            p.requires_grad = not freeze

        # If embeddings are NOT shared, freeze encoder's private embedding
        if self.shared_emb is None:
            for p in self.encoder.tok_emb.parameters():
                p.requires_grad = not freeze
        else:
            # If shared, ensure decoder can train the vocabulary embeddings & LM head
            self.shared_emb.weight.requires_grad = True

        # Decoder LM head must always remain trainable in Stage 2
        self.decoder.lm_head.weight.requires_grad = True

        if freeze:
            self.encoder.eval()
            self.predictor.eval()
        else:
            self.encoder.train()
            self.predictor.train()

    def forward_lejepa(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 1 Forward Pass:
        Trains the continuous planner strictly with L2 prediction loss + SIGReg.

        tokens: [B, T]
        Returns:
            total_loss: L2_pred + lambda * SIGReg
            pred_loss:  MSE(z_hat_{t+1}, stop_grad(z_{t+1}))
            sigreg_loss: Epps-Pulley isotropic Gaussian statistic
            z_hat:      Predicted future latents [B, T-1, d_latent]
            z_target:   Ground truth future latents [B, T-1, d_latent]
        """
        # Encode full sequence causally: [B, T, d_latent]
        z_all, _ = self.encoder(tokens)

        # Context latents: z_1 to z_{T-1}
        z_ctx = z_all[:, :-1, :]

        # Ground truth target latents: z_2 to z_T with stop-gradient
        z_target = z_all[:, 1:, :].detach()

        # Predict next latent state: z_hat_{t+1} given z_{\le t}
        z_hat = self.predictor(z_ctx)

        # 1. L2 Prediction Loss in continuous latent space
        pred_loss = F.mse_loss(z_hat, z_target)

        # 2. SIGReg Regularization Loss (ensures isotropic Gaussian, prevents collapse)
        # Pass (T, B, D) or (1, N, D)
        sigreg_loss = self.sigreg(z_all.transpose(0, 1))

        # Total Stage 1 Loss
        total_loss = pred_loss + self.sigreg_weight * sigreg_loss
        return total_loss, pred_loss, sigreg_loss, z_hat, z_target

    def forward_decoder(
        self,
        tokens_in: torch.Tensor,
        z_hat_next: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2 Forward Pass:
        Trains the conditional decoder with frozen or stop-grad predicted latents.

        tokens_in:  [B, T-1] - Input tokens x_{\le t}
        z_hat_next: [B, T-1, d_latent] - Predicted latents \hat{z}_{t+1}
        targets:    [B, T-1] - Ground-truth next tokens x_{t+1}
        """
        # Ensure stop-gradient on predicted latents to keep latent space isotropic
        z_hat_cond = z_hat_next.detach()

        # Decoder computes P(x_{t+1} | x_{\le t}, \hat{z}_{t+1})
        logits = self.decoder(tokens_in, z_hat_cond)

        # Cross-Entropy loss over discrete vocabulary
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss, logits

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        stage: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        General forward pass supporting Stage 1, Stage 2, or End-to-End evaluation.
        tokens:  [B, T]
        targets: [B, T-1] (optional, default slices tokens[:, 1:])
        """
        if targets is None and tokens.size(1) > 1:
            targets = tokens[:, 1:]
            tokens_in = tokens[:, :-1]
        else:
            tokens_in = tokens[:, :-1]

        res = {}
        if stage in ("1", "lejepa"):
            loss, pred_loss, sigreg_loss, z_hat, z_target = self.forward_lejepa(tokens)
            res["loss"] = loss
            res["pred_loss"] = pred_loss
            res["sigreg_loss"] = sigreg_loss
            res["z_hat"] = z_hat
            res["z_target"] = z_target
            return res

        # Obtain predicted latents from LeJEPA
        if stage in ("2", "decoder"):
            with torch.no_grad():
                z_all, _ = self.encoder(tokens)
                z_ctx = z_all[:, :-1, :]
                z_hat = self.predictor(z_ctx)
            ce_loss, logits = self.forward_decoder(tokens_in, z_hat, targets)
            res["loss"] = ce_loss
            res["ce_loss"] = ce_loss
            res["logits"] = logits
            return res

        # Both stages computed (for end-to-end training/evaluation)
        jepa_loss, pred_loss, sigreg_loss, z_hat, z_target = self.forward_lejepa(tokens)
        ce_loss, logits = self.forward_decoder(tokens_in, z_hat, targets)

        res["loss"] = ce_loss + jepa_loss
        res["ce_loss"] = ce_loss
        res["pred_loss"] = pred_loss
        res["sigreg_loss"] = sigreg_loss
        res["logits"] = logits
        return res

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation step-by-step:
        1. LeJEPA encodes x_{\le t} and predicts next latent plan \hat{z}_{t+1}.
        2. Decoder generates token distribution P(x_{t+1} | x_{\le t}, \hat{z}_{t+1}).
        3. Samples next token x_{t+1} and appends to history.
        """
        self.eval()
        tokens = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            # 1. LeJEPA encodes full context
            z_all, _ = self.encoder(tokens)

            # Predict next latent state \hat{z}_{next} for the current last position
            z_hat = self.predictor(z_all)  # [B, T, d_latent]

            # 2. Condition decoder on tokens and predicted next latents
            logits = self.decoder(tokens, z_hat)  # [B, T, vocab_size]
            next_token_logits = logits[:, -1, :]  # Logits for next token

            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


class StandardTransformerLM(nn.Module):
    """
    Standard Autoregressive Language Model Transformer baseline:
        P(x_{t+1} | x_{\le t})
    Used for direct empirical comparison against the LeJEPA Hierarchical Architecture.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        # Standard transformer weight initialization (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        total = sum(p.numel() for p in self.parameters())
        return {"total_params": total}

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if targets is None and tokens.size(1) > 1:
            targets = tokens[:, 1:]
            tokens_in = tokens[:, :-1]
        else:
            tokens_in = tokens[:, :-1]

        x = self.tok_emb(tokens_in)
        for block in self.blocks:
            x = block(x, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x)

        res = {"logits": logits}
        if targets is not None:
            res["loss"] = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            res["ce_loss"] = res["loss"]
        return res

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.eval()
        tokens = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            x = self.tok_emb(tokens)
            for block in self.blocks:
                x = block(x, is_causal=True)
            x = self.norm(x)
            logits = self.lm_head(x)
            next_token_logits = logits[:, -1, :]

            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
