"""
End-to-end integration tests for FirstPrinciplesActiveInferenceAgent and live pixel Gymnasium environment.
"""

import os
import tempfile
import unittest
import numpy as np
import torch
from firstPAgent.agent import FirstPrinciplesActiveInferenceAgent
from firstPAgent.buffer import PixelReplayBuffer
from firstPAgent.env import make_pixel_env


class TestAgentAndIntegration(unittest.TestCase):

    def test_agent_train_step(self):
        """Verify single train_step() update executing blueprint loss calculation."""
        B = 64
        latent_dim = 32
        act_dim = 2
        agent = FirstPrinciplesActiveInferenceAgent(
            in_channels=3,
            latent_dim=latent_dim,
            act_dim=act_dim,
            num_models=3,
            hidden_dim=64,
            horizon=4,
            num_candidates=32,
            num_elites=8,
            device="cpu",
        )

        obs_t = torch.rand(B, 3, 64, 64)
        actions = torch.randn(B, act_dim)
        rewards = torch.randn(B, 1)
        obs_next = torch.rand(B, 3, 64, 64)
        dones = torch.zeros(B, 1)
        dones[::10] = 1.0  # Some terminations

        batch = (obs_t, actions, rewards, obs_next, dones)
        metrics = agent.train_step(batch)

        self.assertIn("loss_wm", metrics)
        self.assertIn("loss_dyn", metrics)
        self.assertIn("loss_reg", metrics)
        self.assertIn("loss_viab", metrics)
        self.assertIn("loss_rew", metrics)
        self.assertIn("std_mean", metrics)
        self.assertIn("off_diag_mean", metrics)
        self.assertIn("batch_epi_unc", metrics)
        self.assertIn("viab_acc", metrics)

        # All losses must be finite
        for k, v in metrics.items():
            self.assertTrue(np.isfinite(v), f"Metric {k} is not finite: {v}")

    def test_live_gym_pixel_environment_rollout(self):
        """Verify live interaction with Gymnasium pixel environment."""
        env = make_pixel_env(env_id="Pendulum-v1", img_size=(64, 64))
        obs_shape = env.observation_space.shape
        act_dim = env.action_space.shape[0]

        agent = FirstPrinciplesActiveInferenceAgent(
            in_channels=obs_shape[0],
            latent_dim=32,
            act_dim=act_dim,
            num_models=3,
            hidden_dim=64,
            horizon=4,
            num_candidates=32,
            num_elites=8,
            device="cpu",
        )

        buffer = PixelReplayBuffer(
            capacity=100,
            obs_shape=obs_shape,
            act_dim=act_dim,
            device="cpu",
        )

        obs, _ = env.reset(seed=42)
        self.assertEqual(obs.shape, (3, 64, 64))

        # Step live environment
        for step in range(5):
            action, plan_diag = agent.select_action(obs, warm_start=True)
            self.assertEqual(action.shape, (act_dim,))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, next_obs, done)
            obs = next_obs
            if done:
                obs, _ = env.reset()

        self.assertEqual(len(buffer), 5)

        # Test checkpoint save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_agent.pt")
            agent.save(ckpt_path)
            self.assertTrue(os.path.exists(ckpt_path))

            agent_reloaded = FirstPrinciplesActiveInferenceAgent(
                in_channels=obs_shape[0],
                latent_dim=32,
                act_dim=act_dim,
                num_models=3,
                hidden_dim=64,
                horizon=4,
                num_candidates=32,
                num_elites=8,
                device="cpu",
            )
            agent_reloaded.load(ckpt_path)
            # Verify inference works with loaded checkpoint
            act_loaded, _ = agent_reloaded.select_action(obs)
            self.assertEqual(act_loaded.shape, (act_dim,))

        env.close()


if __name__ == "__main__":
    unittest.main()
