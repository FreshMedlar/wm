"""
Unit tests for SIGReg loss and latent health metrics.
"""

import unittest
import torch
import torch.nn.functional as F
from firstPAgent.losses import compute_latent_metrics, sigreg_loss


class TestLosses(unittest.TestCase):

    def test_sigreg_identity_covariance(self):
        """When representations have variance 1 and zero correlation, SIGReg loss should be minimal."""
        torch.manual_seed(42)
        B, D = 1000, 32
        z = torch.randn(B, D)
        loss = sigreg_loss(z, alpha=1.0, beta=0.05)
        self.assertLess(loss.item(), 0.15)

    def test_sigreg_collapsed_variance(self):
        """When representations collapse (zero variance), loss_std must be ~1.0."""
        B, D = 128, 32
        z = torch.ones(B, D) * 5.0
        loss = sigreg_loss(z, alpha=1.0, beta=0.0)
        self.assertAlmostEqual(loss.item(), 1.0, delta=0.05)

    def test_sigreg_correlated_dimensions(self):
        """When dimensions are perfectly correlated, loss_cov must be positive and penalize redundancy."""
        B, D = 128, 16
        v = torch.randn(B, 1)
        z = v.repeat(1, D)  # All D dimensions are identical
        loss_uncorrelated = sigreg_loss(torch.randn(B, D), alpha=0.0, beta=1.0)
        loss_correlated = sigreg_loss(z, alpha=0.0, beta=1.0)
        self.assertGreater(loss_correlated.item(), loss_uncorrelated.item())

    def test_compute_latent_metrics(self):
        """Verify diagnostic metrics computation."""
        torch.manual_seed(42)
        B, D = 256, 16
        z = torch.randn(B, D)
        metrics = compute_latent_metrics(z)
        self.assertIn("std_mean", metrics)
        self.assertIn("off_diag_mean", metrics)
        self.assertIn("off_diag_max", metrics)
        self.assertTrue(0.8 <= metrics["std_mean"] <= 1.2)
        self.assertLess(metrics["off_diag_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
