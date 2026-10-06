"""
Unit tests for neural network architectures:
- PixelEncoder
- TransitionPredictor
- PredictorEnsemble
- ViabilityClassifier
"""

import unittest
import torch
from firstPAgent.networks import (
    PixelEncoder,
    PredictorEnsemble,
    TransitionPredictor,
    ViabilityClassifier,
)


class TestNetworks(unittest.TestCase):

    def test_pixel_encoder_shapes(self):
        """PixelEncoder should handle arbitrary resolutions (64x64, 84x84) and produce (B, D)."""
        encoder = PixelEncoder(in_channels=3, latent_dim=64)

        # Resolution 64x64
        x64 = torch.randn(4, 3, 64, 64)
        z64 = encoder(x64)
        self.assertEqual(z64.shape, (4, 64))

        # Resolution 84x84
        x84 = torch.randn(2, 3, 84, 84)
        z84 = encoder(x84)
        self.assertEqual(z84.shape, (2, 64))

    def test_transition_predictor(self):
        """TransitionPredictor should map (z, a) -> next_z."""
        predictor = TransitionPredictor(latent_dim=32, act_dim=2, hidden_dim=64, residual=True)
        z = torch.randn(8, 32)
        a = torch.randn(8, 2)
        next_z = predictor(z, a)
        self.assertEqual(next_z.shape, (8, 32))

    def test_predictor_ensemble_uncertainty(self):
        """PredictorEnsemble should quantify epistemic uncertainty as prediction variance."""
        ensemble = PredictorEnsemble(
            num_models=5, latent_dim=32, act_dim=2, hidden_dim=64, residual=True
        )
        z = torch.randn(10, 32)
        a = torch.randn(10, 2)

        mean_pred, uncertainty, all_preds = ensemble.predict_with_uncertainty(z, a)

        self.assertEqual(mean_pred.shape, (10, 32))
        self.assertEqual(uncertainty.shape, (10,))
        self.assertEqual(all_preds.shape, (5, 10, 32))
        self.assertTrue((uncertainty >= 0.0).all())

    def test_viability_classifier(self):
        """ViabilityClassifier outputs in [0, 1] and computes costs."""
        viab = ViabilityClassifier(latent_dim=32, hidden_dim=64)
        z = torch.randn(8, 32)
        out = viab(z)
        self.assertEqual(out.shape, (8, 1))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

        cost_log = viab.compute_viability_cost(z, mode="log")
        cost_linear = viab.compute_viability_cost(z, mode="linear")
        self.assertEqual(cost_log.shape, (8,))
        self.assertEqual(cost_linear.shape, (8,))
        self.assertTrue((cost_log >= 0.0).all())

    def test_reward_predictor(self):
        """RewardPredictor should map (z, a) -> scalar reward."""
        from firstPAgent.networks import RewardPredictor

        rew_pred = RewardPredictor(latent_dim=32, act_dim=2, hidden_dim=64)
        z = torch.randn(8, 32)
        a = torch.randn(8, 2)
        r = rew_pred(z, a)
        self.assertEqual(r.shape, (8,))
        self.assertTrue(torch.isfinite(r).all())


if __name__ == "__main__":
    unittest.main()
