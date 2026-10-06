"""
Raw FIFO Sequence Replay Buffer for Pixel Observations.

Stores raw tuples: (obs_t, act_t, rew_t, obs_next, done).
Latents are never cached to ensure fresh representations throughout training.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch


class PixelReplayBuffer:
    """
    Standard FIFO Replay Buffer for raw image observations, continuous actions, and rewards.
    Observations are stored as uint8 (C, H, W) to minimize RAM footprint.
    """

    def __init__(
        self,
        capacity: int = 100000,
        obs_shape: Tuple[int, int, int] = (3, 64, 64),
        act_dim: int = 2,
        context_dim: int = 0,
        device: Union[str, torch.device] = "cpu",
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.device = torch.device(device)

        self.idx = 0
        self.size = 0

        # Memory allocation (uint8 for pixels saves ~75% memory compared to float32)
        self.obs_buffer = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs_buffer = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action_buffer = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype=np.float32)
        if context_dim > 0:
            self.context_buffer = np.zeros((capacity, context_dim), dtype=np.float32)
        else:
            self.context_buffer = None

    def _to_uint8(self, obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert observation to uint8 array."""
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        obs = np.asarray(obs)
        if obs.dtype == np.uint8:
            return obs
        # If float in [0, 1], scale to [0, 255]
        if np.issubdtype(obs.dtype, np.floating):
            obs = np.clip(obs * 255.0, 0.0, 255.0).astype(np.uint8)
        return obs

    def add(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        arg3: Union[float, int, np.ndarray, torch.Tensor],
        arg4: Union[np.ndarray, torch.Tensor, bool, float, int] = None,
        arg5: Optional[Union[bool, float, int]] = None,
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Add a transition tuple to the FIFO buffer.
        """
        if arg5 is not None:
            # 5-argument signature: (obs, action, reward, next_obs, done)
            reward = float(arg3)
            next_obs = arg4
            done = bool(arg5)
        elif arg4 is not None:
            # 4-argument signature: (obs, action, next_obs, done)
            reward = 0.0
            next_obs = arg3
            done = bool(arg4)
        else:
            raise ValueError("PixelReplayBuffer.add() requires at least 4 arguments (obs, action, next_obs, done).")

        obs_u8 = self._to_uint8(obs)
        next_obs_u8 = self._to_uint8(next_obs)

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32)

        self.obs_buffer[self.idx] = obs_u8
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.next_obs_buffer[self.idx] = next_obs_u8
        self.done_buffer[self.idx] = 1.0 if done else 0.0

        if self.context_buffer is not None and context is not None:
            if isinstance(context, torch.Tensor):
                context = context.detach().cpu().numpy()
            self.context_buffer[self.idx] = np.asarray(context, dtype=np.float32).reshape(-1)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: Union[str, torch.device, None] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Sample a random mini-batch of transitions.
        """
        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: size={self.size}, requested={batch_size}"
            )

        indices = np.random.randint(0, self.size, size=batch_size)
        target_device = torch.device(device) if device is not None else self.device

        # Convert uint8 [0, 255] to float32 [0.0, 1.0]
        obs_t = (
            torch.from_numpy(self.obs_buffer[indices]).to(target_device).float() / 255.0
        )
        actions = torch.from_numpy(self.action_buffer[indices]).to(target_device).float()
        rewards = torch.from_numpy(self.reward_buffer[indices]).to(target_device).float()
        obs_next = (
            torch.from_numpy(self.next_obs_buffer[indices]).to(target_device).float() / 255.0
        )
        dones = torch.from_numpy(self.done_buffer[indices]).to(target_device).float()

        if self.context_buffer is not None:
            context = torch.from_numpy(self.context_buffer[indices]).to(target_device).float()
            return obs_t, actions, rewards, obs_next, dones, context

        return obs_t, actions, rewards, obs_next, dones

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_samples: int) -> bool:
        return self.size >= min_samples
