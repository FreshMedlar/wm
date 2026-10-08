"""
Active Inference Controller / Model Predictive Control (MPC) Planner.

Implements trajectory optimization via Cross-Entropy Method (CEM) and Model Predictive
Path Integral (MPPI) to minimize Expected Free Energy (EFE):
  G(π) = - w_epi * Epistemic(σ²) - w_emp * Empowerment(Var_candidates) + w_viab * Viability(ẑ)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from FPagent.networks import PredictorEnsemble, RewardPredictor, ViabilityClassifier


class ActiveInferencePlanner:
    """
    Trajectory optimizer using CEM or MPPI over latent state predictions.
    Supports Expected Free Energy (EFE) with Pragmatic Value (Reward),
    Viability Boundary (Crash avoidance), Epistemic Value (Curiosity), and Empowerment.
    """

    def __init__(
        self,
        predictor_ensemble: PredictorEnsemble,
        viability_net: ViabilityClassifier,
        reward_net: Optional[RewardPredictor] = None,
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
        temperature: float = 0.5,
        device: Union[str, torch.device] = "cpu",
    ):
        self.predictor_ensemble = predictor_ensemble
        self.viability_net = viability_net
        self.reward_net = reward_net
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = min(num_elites, num_candidates)
        self.num_iterations = num_iterations
        self.w_rew = w_rew
        self.w_epi = w_epi
        self.w_viab = w_viab
        self.w_emp = w_emp
        self.viab_cost_mode = viab_cost_mode
        self.action_low = action_low
        self.action_high = action_high
        self.planner_type = planner_type.lower()
        self.temperature = temperature
        self.device = torch.device(device)

        self.act_dim = predictor_ensemble.act_dim
        self.latent_dim = predictor_ensemble.latent_dim

        # Previous mean action sequence for warm-starting
        self.prev_mean: Optional[torch.Tensor] = None

    def reset(self):
        """Reset warm-started trajectory distribution between episodes."""
        self.prev_mean = None

    @torch.no_grad()
    def plan(
        self,
        z_t: torch.Tensor,
        warm_start: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Execute trajectory optimization starting from current latent state z_t.

        Args:
            z_t: Current active latent representation (1, D) or (D,)
            warm_start: Whether to shift previous solution for initial distribution

        Returns:
            best_action: (act_dim,) numpy array to execute in the environment.
            diagnostics: Dict containing cost breakdowns and statistics.
        """
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)  # (1, D)
        z_t = z_t.to(self.device)

        # 1. Initialize distribution parameters: mu, sigma: (H, act_dim)
        if warm_start and self.prev_mean is not None:
            # Shift previous solution forward by 1 step
            mean = torch.zeros((self.horizon, self.act_dim), device=self.device)
            mean[:-1] = self.prev_mean[1:]
            mean[-1] = 0.0
        else:
            mean = torch.zeros((self.horizon, self.act_dim), device=self.device)

        std = torch.ones((self.horizon, self.act_dim), device=self.device)

        best_actions = None
        best_cost = float("inf")
        diagnostics = {}

        # 2. Optimization iterations
        for iteration in range(self.num_iterations):
            # Sample candidate action sequences: (N, H, act_dim)
            noise = torch.randn(
                (self.num_candidates, self.horizon, self.act_dim), device=self.device
            )
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates = torch.clamp(candidates, self.action_low, self.action_high)

            # Evaluate candidate trajectories in latent space
            costs, cost_breakdown = self._evaluate_candidates(z_t, candidates)

            # Update distribution
            if self.planner_type == "cem":
                # Elite selection: top-K lowest cost
                elite_idx = torch.topk(costs, k=self.num_elites, largest=False).indices
                elites = candidates[elite_idx]  # (K, H, act_dim)

                new_mean = elites.mean(dim=0)
                new_std = elites.std(dim=0) + 1e-4

                # Soft update / momentum
                alpha = 0.2 if iteration > 0 else 0.0
                mean = alpha * mean + (1.0 - alpha) * new_mean
                std = alpha * std + (1.0 - alpha) * new_std

                iter_best_idx = elite_idx[0]
                iter_best_cost = costs[iter_best_idx].item()
                if iter_best_cost < best_cost:
                    best_cost = iter_best_cost
                    best_actions = candidates[iter_best_idx]

            elif self.planner_type == "mppi":
                # Softmax path integral weighting
                min_cost = costs.min()
                weights = torch.softmax(-(costs - min_cost) / self.temperature, dim=0)  # (N,)
                # Weighted mean: sum_n w_n * a_n
                mean = (weights.view(-1, 1, 1) * candidates).sum(dim=0)
                # Weighted std
                diff = candidates - mean.unsqueeze(0)
                new_std = torch.sqrt((weights.view(-1, 1, 1) * diff.pow(2)).sum(dim=0) + 1e-4)
                std = 0.5 * std + 0.5 * new_std

                iter_best_idx = torch.argmin(costs)
                iter_best_cost = costs[iter_best_idx].item()
                if iter_best_cost < best_cost:
                    best_cost = iter_best_cost
                    best_actions = candidates[iter_best_idx]

            diagnostics = cost_breakdown
            diagnostics["best_cost"] = best_cost

        # Cache mean for warm-starting next step
        self.prev_mean = mean.detach().clone()

        # Action to execute: first step of optimized sequence
        chosen_action = mean[0].detach().cpu().numpy()
        chosen_action = np.clip(chosen_action, self.action_low, self.action_high)

        return chosen_action, diagnostics

    def _evaluate_candidates(
        self,
        z_init: torch.Tensor,
        candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Roll out candidate trajectories in latent space and compute EFE cost.

        Args:
            z_init: (1, D)
            candidates: (N, H, act_dim)

        Returns:
            costs: (N,) scalar cost for each candidate sequence
            breakdown: Dict of average cost components
        """
        N, H, _ = candidates.shape
        # Expand initial state for all N candidates
        z_curr = z_init.repeat(N, 1)  # (N, D)

        cum_viab_cost = torch.zeros(N, device=self.device)
        cum_epi_cost = torch.zeros(N, device=self.device)
        cum_reward = torch.zeros(N, device=self.device)

        for step in range(H):
            a_step = candidates[:, step, :]  # (N, act_dim)

            # 1. Extrinsic reward prediction (Pragmatic Value)
            if self.w_rew > 0 and self.reward_net is not None:
                r_step = self.reward_net(z_curr, a_step)
                cum_reward += r_step

            # 2. Viability cost of current predicted state
            v_cost = self.viability_net.compute_viability_cost(
                z_curr, mode=self.viab_cost_mode
            )
            cum_viab_cost += v_cost

            # 3. Ensemble forward step and epistemic uncertainty
            z_next_mean, uncertainty, _ = (
                self.predictor_ensemble.predict_with_uncertainty(z_curr, a_step)
            )
            cum_epi_cost += uncertainty

            # Update current state for candidate rollouts
            z_curr = z_next_mean

        # 4. Terminal Empowerment: Dispersion of reachable states across candidates at horizon H
        # Var_candidates(ẑ_{t+H})
        z_terminal = z_curr  # (N, D)
        z_term_mean = z_terminal.mean(dim=0, keepdim=True)  # (1, D)
        # Distance of each candidate from the population center
        cand_dispersion = (z_terminal - z_term_mean).pow(2).sum(dim=-1) / self.latent_dim  # (N,)

        # Overall trajectory cost G(π):
        # C^{(n)} = - w_rew * Reward + sum_{τ} (w_viab * Viab - w_epi * Epistemic) - w_emp * Empowerment
        total_costs = (
            - self.w_rew * cum_reward
            + self.w_viab * cum_viab_cost
            - self.w_epi * cum_epi_cost
            - self.w_emp * cand_dispersion
        )

        breakdown = {
            "mean_reward_pred": cum_reward.mean().item(),
            "mean_viab_cost": cum_viab_cost.mean().item(),
            "mean_epi_value": cum_epi_cost.mean().item(),
            "mean_empowerment": cand_dispersion.mean().item(),
            "mean_total_cost": total_costs.mean().item(),
        }

        return total_costs, breakdown
