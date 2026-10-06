"""
First-Principles Active Inference Agent (Single Shared Encoder with Target Detach).

Orchestrates:
- Single Perceptual Encoder E_θ(·) with SIGReg regularizer
- Causal Ensemble World Model {P_ϕ1, ..., P_ϕM}
- Viability Network V_ψ(·)
- Active Inference Trajectory Optimizer (CEM / MPPI)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from firstPAgent.losses import compute_latent_metrics, sigreg_loss
from firstPAgent.networks import (
    PixelEncoder,
    PredictorEnsemble,
    RewardPredictor,
    ViabilityClassifier,
)
from firstPAgent.planner import ActiveInferencePlanner


class FirstPrinciplesActiveInferenceAgent:
    """
    Active Inference Agent operating directly on raw pixels.
    Uses a single shared encoder without EMA target networks, preventing representation
    collapse via SIGReg and stop-gradient on target latents.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        act_dim: int = 2,
        context_dim: int = 0,
        num_models: int = 5,
        hidden_dim: int = 256,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        gamma: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        horizon: int = 12,
        num_candidates: int = 256,
        num_elites: int = 32,
        num_iterations: int = 3,
        w_rew: float = 0.0,
        w_epi: float = 1.0,
        w_viab: float = 10.0,
        w_emp: float = 0.5,
        viab_cost_mode: str = "log",
        action_low: float = -1.0,
        action_high: float = 1.0,
        planner_type: str = "cem",
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.gamma = gamma
        self.w_rew = w_rew

        # 0. Sketch Isotropic Gaussian Regularizer (True SIGReg)
        from firstPAgent.losses import SIGReg
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj).to(self.device)

        # 1. Single Perceptual Encoder E_θ
        self.encoder = PixelEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
        ).to(self.device)

        # 2. Causal Ensemble World Model {P_ϕ1, ..., P_ϕM}
        self.predictor_ensemble = PredictorEnsemble(
            num_models=num_models,
            latent_dim=latent_dim,
            act_dim=act_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            residual=True,
        ).to(self.device)

        # 3. Viability Network V_ψ
        self.viability_net = ViabilityClassifier(
            latent_dim=latent_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim // 2,
        ).to(self.device)

        # 4. Extrinsic Reward Predictor R_ω (Pragmatic Value)
        self.reward_net = RewardPredictor(
            latent_dim=latent_dim,
            act_dim=act_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim // 2,
        ).to(self.device)

        # 5. Active Inference Controller / MPC Planner
        self.planner = ActiveInferencePlanner(
            predictor_ensemble=self.predictor_ensemble,
            viability_net=self.viability_net,
            reward_net=self.reward_net,
            horizon=horizon,
            num_candidates=num_candidates,
            num_elites=num_elites,
            num_iterations=num_iterations,
            w_rew=w_rew,
            w_epi=w_epi,
            w_viab=w_viab,
            w_emp=w_emp,
            viab_cost_mode=viab_cost_mode,
            action_low=action_low,
            action_high=action_high,
            planner_type=planner_type,
            device=self.device,
        )

        # 6. Optimizers
        # Joint optimizer for World Model (Encoder + Predictor Heads)
        wm_params = list(self.encoder.parameters()) + list(self.predictor_ensemble.parameters())
        self.opt_world_model = torch.optim.AdamW(
            wm_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Independent optimizer for Viability Classifier
        self.opt_viability = torch.optim.AdamW(
            self.viability_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Independent optimizer for Reward Predictor
        self.opt_reward = torch.optim.AdamW(
            self.reward_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_step(
        self,
        batch: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        gamma: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Executes one joint gradient update step on the world model, viability network, and reward head.

        Args:
            batch: Tuple of either:
                   (obs_t, actions, obs_next, dones) or
                   (obs_t, actions, rewards, obs_next, dones)
            gamma: Regularization weight override (defaults to self.gamma).

        Returns:
            Dict containing detailed training metrics and diagnostic stats.
        """
        if gamma is None:
            gamma = self.gamma

        context = None
        if len(batch) == 6:
            obs_t, actions, rewards, obs_next, dones, context = batch
            rewards = rewards.to(self.device)
            context = context.to(self.device)
        elif len(batch) == 5:
            obs_t, actions, rewards, obs_next, dones = batch
            rewards = rewards.to(self.device)
        else:
            obs_t, actions, obs_next, dones = batch
            rewards = torch.zeros((actions.size(0), 1), device=self.device)

        obs_t = obs_t.to(self.device)
        actions = actions.to(self.device)
        obs_next = obs_next.to(self.device)
        dones = dones.to(self.device)

        # 1. Forward through SINGLE shared encoder
        z_t = self.encoder(obs_t)
        z_next = self.encoder(obs_next)

        # 2. Prevent target-pulling via stop-gradient (CRITICAL STEP)
        z_next_target = z_next.detach()

        # 3. Predictor dynamics loss
        pred_losses = []
        for predictor in self.predictor_ensemble:
            z_next_pred = predictor(z_t, actions, context=context)
            pred_losses.append(F.mse_loss(z_next_pred, z_next_target))
        loss_dyn = torch.stack(pred_losses).mean()

        # 4. Anti-collapse regularizer (True SIGReg)
        loss_reg = self.sigreg(z_t)

        # 5. Combined World Model backprop
        loss_wm = loss_dyn + gamma * loss_reg
        self.opt_world_model.zero_grad(set_to_none=True)
        loss_wm.backward()
        self.opt_world_model.step()

        # 6. Viability classifier update (independent loss, detached latents)
        v_pred = self.viability_net(z_t.detach(), context=context)
        loss_viab = F.binary_cross_entropy(v_pred, dones)
        self.opt_viability.zero_grad(set_to_none=True)
        loss_viab.backward()
        self.opt_viability.step()

        # 7. Extrinsic Reward Predictor update (Huber/Smooth L1 loss on detached latents)
        # Smooth L1 prevents gradient shocks from sparse, high-magnitude terminal rewards (e.g. -100 crash)
        r_pred = self.reward_net(z_t.detach(), actions, context=context)
        loss_rew = F.smooth_l1_loss(r_pred, rewards.view(-1))
        self.opt_reward.zero_grad(set_to_none=True)
        loss_rew.backward()
        self.opt_reward.step()

        # 8. Diagnostics (Section 5 Developer Checklist)
        latent_health = compute_latent_metrics(z_t)

        with torch.no_grad():
            # Epistemic uncertainty on the training batch
            _, batch_epi_unc, _ = self.predictor_ensemble.predict_with_uncertainty(
                z_t.detach(), actions, context=context
            )
            mean_epi_unc = batch_epi_unc.mean().item()

            # Viability classification accuracy
            viab_binary_preds = (v_pred >= 0.5).float()
            viab_acc = (viab_binary_preds == dones).float().mean().item()

        return {
            "loss_wm": loss_wm.item(),
            "loss_dyn": loss_dyn.item(),
            "loss_reg": loss_reg.item(),
            "loss_viab": loss_viab.item(),
            "loss_rew": loss_rew.item(),
            "std_mean": latent_health["std_mean"],
            "off_diag_mean": latent_health["off_diag_mean"],
            "off_diag_max": latent_health["off_diag_max"],
            "batch_epi_unc": mean_epi_unc,
            "viab_acc": viab_acc,
        }

    @torch.no_grad()
    def select_action(
        self,
        obs: Union[torch.Tensor, np.ndarray],
        context: Optional[torch.Tensor] = None,
        warm_start: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Encode raw pixel observation and plan action using Active Inference.

        Args:
            obs: Observation frame (C, H, W) or (1, C, H, W)
            context: Optional language context vector (1, context_dim) or (context_dim,)
            warm_start: Shift previous trajectory solution

        Returns:
            action: (act_dim,) numpy array
            plan_diagnostics: Dict containing planning costs and details
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)  # (1, C, H, W)
        obs = obs.to(self.device)

        # Encode current pixel frame
        z_t = self.encoder(obs)

        # Plan action sequence conditioned on optional language context
        action, plan_diagnostics = self.planner.plan(
            z_t, context=context, warm_start=warm_start
        )
        return action, plan_diagnostics

    def reset_planner(self):
        """Call at episode boundaries to reset planner state."""
        self.planner.reset()

    def save(self, filepath: str):
        """Save model checkpoints."""
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "predictor_ensemble": self.predictor_ensemble.state_dict(),
                "viability_net": self.viability_net.state_dict(),
                "reward_net": self.reward_net.state_dict(),
                "opt_world_model": self.opt_world_model.state_dict(),
                "opt_viability": self.opt_viability.state_dict(),
                "opt_reward": self.opt_reward.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.predictor_ensemble.load_state_dict(checkpoint["predictor_ensemble"])
        self.viability_net.load_state_dict(checkpoint["viability_net"])
        if "reward_net" in checkpoint:
            self.reward_net.load_state_dict(checkpoint["reward_net"])
        self.opt_world_model.load_state_dict(checkpoint["opt_world_model"])
        self.opt_viability.load_state_dict(checkpoint["opt_viability"])
        if "opt_reward" in checkpoint:
            self.opt_reward.load_state_dict(checkpoint["opt_reward"])
