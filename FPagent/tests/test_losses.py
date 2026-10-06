"""
Unit tests for SIGReg loss and latent health metrics.
"""

import unittest
import torch
import torch.nn.functional as F
from firstPAgent.losses import compute_latent_metrics, sigreg_loss


class TestLosses(unittest.TestCase):

    def test_sigreg_standard_normal(self):
        """When representations are drawn from N(0, I), SIGReg Epps-Pulley statistic should be minimal."""
        torch.manual_seed(42)
        B, D = 500, 32
        z = torch.randn(B, D)
        loss = sigreg_loss(z, knots=17, num_proj=512)
        # For standard normal, the statistic approaches 0 (subject to finite sample error)
        self.assertLess(loss.item(), 5.0)

    def test_sigreg_collapsed_distribution(self):
        """When representations collapse to a constant, SIGReg statistic should be significantly higher."""
        torch.manual_seed(42)
        B, D = 128, 32
        z_normal = torch.randn(B, D)
        z_collapsed = torch.zeros(B, D)
        loss_normal = sigreg_loss(z_normal, knots=17, num_proj=512)
        loss_collapsed = sigreg_loss(z_collapsed, knots=17, num_proj=512)
        self.assertGreater(loss_collapsed.item(), loss_normal.item() * 2)

    def test_sigreg_gradient_flow(self):
        """Verify gradients flow from SIGReg back to latent representations."""
        z = torch.randn(64, 32, requires_grad=True)
        loss = sigreg_loss(z, knots=17, num_proj=256)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.any(z.grad != 0.0))

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
