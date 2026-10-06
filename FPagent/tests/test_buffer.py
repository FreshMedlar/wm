"""
Unit tests for Raw FIFO Replay Buffer.
"""

import unittest
import numpy as np
import torch
from firstPAgent.buffer import PixelReplayBuffer


class TestBuffer(unittest.TestCase):

    def test_buffer_add_and_fifo_eviction(self):
        """Verify transitions storage and FIFO overwriting at capacity."""
        capacity = 10
        buffer = PixelReplayBuffer(capacity=capacity, obs_shape=(3, 32, 32), act_dim=2)

        # Add 15 transitions with 5 arguments (including reward)
        for i in range(15):
            obs = np.ones((3, 32, 32), dtype=np.float32) * (i / 15.0)
            action = np.array([float(i), -float(i)], dtype=np.float32)
            reward = float(i) * 2.5
            next_obs = obs + 0.01
            done = (i % 5 == 0)
            buffer.add(obs, action, reward, next_obs, done)

        self.assertEqual(len(buffer), capacity)  # Max capacity capped at 10
        self.assertEqual(buffer.size, 10)

        # Sample batch
        batch_size = 4
        obs_t, actions, rewards, obs_next, dones = buffer.sample(batch_size)

        self.assertEqual(obs_t.shape, (batch_size, 3, 32, 32))
        self.assertEqual(actions.shape, (batch_size, 2))
        self.assertEqual(rewards.shape, (batch_size, 1))
        self.assertEqual(obs_next.shape, (batch_size, 3, 32, 32))
        self.assertEqual(dones.shape, (batch_size, 1))

        # Pixel values must be in [0.0, 1.0]
        self.assertTrue(obs_t.min() >= 0.0 and obs_t.max() <= 1.0)
        self.assertTrue(obs_next.min() >= 0.0 and obs_next.max() <= 1.0)

    def test_buffer_backward_compatibility_4_args(self):
        """Verify that 4-argument add(obs, action, next_obs, done) defaults reward to 0.0."""
        buffer = PixelReplayBuffer(capacity=5, obs_shape=(3, 16, 16), act_dim=1)
        obs = np.zeros((3, 16, 16), dtype=np.float32)
        action = np.zeros(1, dtype=np.float32)
        next_obs = np.ones((3, 16, 16), dtype=np.float32)
        buffer.add(obs, action, next_obs, False)

        _, _, rewards, _, _ = buffer.sample(1)
        self.assertEqual(rewards.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
