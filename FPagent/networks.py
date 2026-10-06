"""
Neural Network Architectures for First-Principles Active Inference Agent.

Includes:
- PixelEncoder: Single shared convolutional perceptual encoder E_θ(o_t).
- TransitionPredictor: Transition dynamics predictor P_ϕ(z_t, a_t).
- PredictorEnsemble: Causal ensemble world model {P_ϕ1, ..., P_ϕM}.
- ViabilityClassifier: Auxiliary homeostatic classifier V_ψ(z_t) -> [0, 1].
- RewardPredictor: Auxiliary pragmatic reward predictor R_ω(z_t, a_t) -> R.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelEncoder(nn.Module):
    """
    Single Perceptual Encoder E_θ(·) mapping raw pixel frames to latent representations.

    Input:
        Image observations tensor of shape (B, C, H, W) normalized to [0, 1].
    Output:
        Active latent representation z of shape (B, D).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            # (B, in_channels, H, W) -> (B, base_channels, H/2, W/2)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # -> (B, base_channels * 2, H/4, W/4)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # -> (B, base_channels * 4, H/8, W/8)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # -> (B, base_channels * 8, H/16, W/16)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # Robust spatial pooling to decouple from exact input resolution (e.g. 64x64 or 84x84)
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        flatten_dim = (base_channels * 8) * 2 * 2  # 256 * 4 = 1024

        # Linear projection to latent space D
        # Note: No hard LayerNorm at the final projection so the latent space can freely
        # scale its variance to std ~ 1.0 under SIGReg regularization.
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through single encoder.
        obs: (B, C, H, W)
        returns: (B, D)
        """
        features = self.conv(obs)
        flattened = features.reshape(features.size(0), -1)
        z = self.fc(flattened)
        return z


class TransitionPredictor(nn.Module):
    """
    Transition model P_ϕm(z_t, a_t, [c_t]) predicting the next latent state ẑ_{t+1}.
    Optionally conditioned on language context c_t.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        act_dim: int = 2,
        context_dim: int = 0,
        hidden_dim: int = 256,
        residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.residual = residual

        in_dim = latent_dim + act_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z: (..., D)
        a: (..., A)
        context: optional (..., C) context vector
        returns: (..., D)
        """
        inputs = [z, a]
        if self.context_dim > 0 and context is not None:
            # Broadcast context to match leading dimensions of z
            if context.shape[:-1] != z.shape[:-1]:
                target_shape = list(z.shape[:-1]) + [self.context_dim]
                context = context.reshape(-1, self.context_dim)
                if context.size(0) == 1:
                    context = context.expand(*target_shape)
                else:
                    context = context.expand(*target_shape)
            inputs.append(context)

        x = torch.cat(inputs, dim=-1)
        delta = self.net(x)
        if self.residual:
            return z + delta
        return delta


class PredictorEnsemble(nn.Module):
    """
    Causal Ensemble World Model {P_ϕ1, ..., P_ϕM}.
    Provides independent transition models and epistemic uncertainty quantification.
    Supports optional language context conditioning.
    """

    def __init__(
        self,
        num_models: int = 5,
        latent_dim: int = 64,
        act_dim: int = 2,
        context_dim: int = 0,
        hidden_dim: int = 256,
        residual: bool = True,
    ):
        super().__init__()
        self.num_models = num_models
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.context_dim = context_dim

        self.predictors = nn.ModuleList([
            TransitionPredictor(
                latent_dim=latent_dim,
                act_dim=act_dim,
                context_dim=context_dim,
                hidden_dim=hidden_dim,
                residual=residual,
            )
            for _ in range(num_models)
        ])

    def __len__(self) -> int:
        return self.num_models

    def __iter__(self):
        return iter(self.predictors)

    def __getitem__(self, idx: int) -> TransitionPredictor:
        return self.predictors[idx]

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate all M models on input (z, a, [context]).
        returns: (M, ..., D)
        """
        preds = [predictor(z, a, context=context) for predictor in self.predictors]
        return torch.stack(preds, dim=0)

    def predict_with_uncertainty(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the ensemble mean prediction and epistemic uncertainty.

        Returns:
            mean_pred: (..., D) ensemble mean next state
            uncertainty: (...) epistemic variance across models
            all_preds: (M, ..., D) raw predictions from all M models
        """
        all_preds = self.forward(z, a, context=context)
        mean_pred = all_preds.mean(dim=0)

        diff = all_preds - mean_pred.unsqueeze(0)
        sq_err = diff.pow(2).sum(dim=-1)
        uncertainty = sq_err.mean(dim=0)

        return mean_pred, uncertainty, all_preds


class ViabilityClassifier(nn.Module):
    """
    Viability Classifier V_ψ(z, [context]) mapping latent state to probability of failure [0, 1].
    Trained on empirical termination flags via BCE.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        context_dim: int = 0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        in_dim = latent_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = [z]
        if self.context_dim > 0 and context is not None:
            if context.shape[:-1] != z.shape[:-1]:
                target_shape = list(z.shape[:-1]) + [self.context_dim]
                context = context.reshape(-1, self.context_dim).expand(*target_shape)
            inputs.append(context)
        x = torch.cat(inputs, dim=-1)
        return self.net(x)

    def compute_viability_cost(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mode: str = "log",
        eps: float = 1e-4,
    ) -> torch.Tensor:
        prob = self.forward(z, context=context).squeeze(-1)
        if mode == "log":
            return -torch.log(torch.clamp(1.0 - prob, min=eps))
        elif mode == "linear":
            return prob
        else:
            raise ValueError(f"Unknown viability cost mode: {mode}")


class RewardPredictor(nn.Module):
    """
    Reward Predictor R_ω(z_t, a_t, [context]) mapping latent state and action to expected reward.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        act_dim: int = 2,
        context_dim: int = 0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.context_dim = context_dim

        in_dim = latent_dim + act_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = [z, a]
        if self.context_dim > 0 and context is not None:
            if context.shape[:-1] != z.shape[:-1]:
                target_shape = list(z.shape[:-1]) + [self.context_dim]
                context = context.reshape(-1, self.context_dim).expand(*target_shape)
            inputs.append(context)
        x = torch.cat(inputs, dim=-1)
        return self.net(x).squeeze(-1)

