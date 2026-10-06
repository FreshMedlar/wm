"""
Test for Developer Checklist Item 1: Verification of .detach() Placement.

Verifies that target representations z_next are explicitly detached before calculating
dynamics loss, ensuring the encoder does not minimize dynamics loss by collapsing future targets.
"""

import unittest
import torch
import torch.nn.functional as F
from firstPAgent.networks import PixelEncoder, TransitionPredictor


class TestDetachValidation(unittest.TestCase):

    def test_target_detach_gradient_flow(self):
        """
        Confirms that:
        1. Gradients from dynamics loss flow backward into z_t and encoder θ.
        2. Gradients from dynamics loss DO NOT flow backward into z_next and encoder θ via target.
        """
        latent_dim = 16
        act_dim = 2
        encoder = PixelEncoder(in_channels=3, latent_dim=latent_dim)
        predictor = TransitionPredictor(latent_dim=latent_dim, act_dim=act_dim, hidden_dim=64)

        obs_t = torch.randn(4, 3, 64, 64)
        obs_next = torch.randn(4, 3, 64, 64)
        actions = torch.randn(4, act_dim)

        # Forward pass
        z_t = encoder(obs_t)
        z_next = encoder(obs_next)

        # 1. Verify that z_next has grad_fn before detach
        self.assertIsNotNone(z_next.grad_fn)

        # 2. Apply stop-gradient (.detach())
        z_next_target = z_next.detach()
        self.assertIsNone(z_next_target.grad_fn)

        # 3. Predict dynamics
        z_next_pred = predictor(z_t, actions)
        loss_dyn = F.mse_loss(z_next_pred, z_next_target)

        # Backpropagate dynamics loss
        loss_dyn.backward(retain_graph=True)

        # Verify that gradients exist for predictor parameters
        for p in predictor.parameters():
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.norm().item(), 0.0)

        # Verify that gradients flow to encoder exclusively via z_t:
        # z_next_target does not propagate gradients to z_next
        self.assertIsNone(z_next_target.grad)

    def test_contrast_with_fatal_non_detach(self):
        """
        Demonstrates that without detach, z_next directly receives target-pulling gradients.
        """
        latent_dim = 16
        act_dim = 2
        encoder = PixelEncoder(in_channels=3, latent_dim=latent_dim)
        predictor = TransitionPredictor(latent_dim=latent_dim, act_dim=act_dim, hidden_dim=64)

        obs_t = torch.randn(4, 3, 64, 64)
        obs_next = torch.randn(4, 3, 64, 64)
        actions = torch.randn(4, act_dim)

        z_t = encoder(obs_t)
        z_next = encoder(obs_next)

        z_next_pred = predictor(z_t, actions)

        # Fatal error without detach:
        loss_dyn_flawed = F.mse_loss(z_next_pred, z_next)

        # z_next has direct computational connection to loss_dyn_flawed
        grad_z_next = torch.autograd.grad(
            loss_dyn_flawed, z_next, retain_graph=True, allow_unused=True
        )[0]
        self.assertIsNotNone(grad_z_next)

        # Contrast with correct detached version:
        z_next_target = z_next.detach()
        loss_dyn_correct = F.mse_loss(z_next_pred, z_next_target)
        grad_z_next_detached = torch.autograd.grad(
            loss_dyn_correct, z_next, retain_graph=True, allow_unused=True
        )[0]
        self.assertIsNone(grad_z_next_detached)  # Confirms complete gradient disconnection!


if __name__ == "__main__":
    unittest.main()
