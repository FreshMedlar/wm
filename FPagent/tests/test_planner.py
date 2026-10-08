"""
Unit tests for Active Inference Controller and MPC Planner.
"""

import unittest
import numpy as np
import torch
from FPagent.networks import PredictorEnsemble, ViabilityClassifier
from FPagent.planner import ActiveInferencePlanner


class TestPlanner(unittest.TestCase):

    def test_planner_cem_execution(self):
        """Verify CEM trajectory optimization over latent dynamics."""
        latent_dim = 16
        act_dim = 2
        ensemble = PredictorEnsemble(num_models=3, latent_dim=latent_dim, act_dim=act_dim, hidden_dim=32)
        viab = ViabilityClassifier(latent_dim=latent_dim, hidden_dim=32)

        planner = ActiveInferencePlanner(
            predictor_ensemble=ensemble,
            viability_net=viab,
            horizon=6,
            num_candidates=32,
            num_elites=8,
            num_iterations=2,
            w_epi=1.0,
            w_viab=5.0,
            w_emp=0.5,
            planner_type="cem",
            action_low=-1.0,
            action_high=1.0,
        )

        z_t = torch.randn(1, latent_dim)
        action, diagnostics = planner.plan(z_t, warm_start=False)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (act_dim,))
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())
        self.assertIn("mean_viab_cost", diagnostics)
        self.assertIn("mean_epi_value", diagnostics)
        self.assertIn("mean_empowerment", diagnostics)
        self.assertIn("mean_total_cost", diagnostics)

    def test_planner_mppi_execution(self):
        """Verify MPPI trajectory optimization over latent dynamics."""
        latent_dim = 16
        act_dim = 2
        ensemble = PredictorEnsemble(num_models=3, latent_dim=latent_dim, act_dim=act_dim, hidden_dim=32)
        viab = ViabilityClassifier(latent_dim=latent_dim, hidden_dim=32)

        planner = ActiveInferencePlanner(
            predictor_ensemble=ensemble,
            viability_net=viab,
            horizon=4,
            num_candidates=32,
            num_iterations=2,
            planner_type="mppi",
        )

        z_t = torch.randn(1, latent_dim)
        action, _ = planner.plan(z_t, warm_start=False)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (act_dim,))

    def test_planner_with_rewards(self):
        """Verify CEM trajectory optimization prioritizing predicted rewards."""
        from FPagent.networks import RewardPredictor

        latent_dim = 16
        act_dim = 2
        ensemble = PredictorEnsemble(num_models=3, latent_dim=latent_dim, act_dim=act_dim, hidden_dim=32)
        viab = ViabilityClassifier(latent_dim=latent_dim, hidden_dim=32)
        rew_net = RewardPredictor(latent_dim=latent_dim, act_dim=act_dim, hidden_dim=32)

        planner = ActiveInferencePlanner(
            predictor_ensemble=ensemble,
            viability_net=viab,
            reward_net=rew_net,
            horizon=4,
            num_candidates=32,
            num_elites=8,
            num_iterations=2,
            w_rew=2.0,
            w_epi=0.5,
            w_viab=5.0,
            w_emp=0.5,
        )

        z_t = torch.randn(1, latent_dim)
        action, diagnostics = planner.plan(z_t, warm_start=False)

        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (act_dim,))
        self.assertIn("mean_reward_pred", diagnostics)


if __name__ == "__main__":
    unittest.main()
