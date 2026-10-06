"""
Gymnasium Pixel Environment Wrappers.

Transforms any Gymnasium continuous control environment into a pixel-observation environment
by intercepting rendered RGB arrays and converting them to normalized (C, H, W) float tensors.
"""

from typing import Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms


class GymPixelWrapper(gym.Wrapper):
    """
    Observation wrapper that converts rendered RGB frames into normalized (C, H, W) float32 tensors.
    """

    def __init__(
        self,
        env: gym.Env,
        img_size: Tuple[int, int] = (64, 64),
        grayscale: bool = False,
    ):
        super().__init__(env)
        self.img_size = img_size
        self.grayscale = grayscale
        self.channels = 1 if grayscale else 3

        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
        ]
        if grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.ToTensor())  # uint8 (H, W, C) -> float32 (C, H, W) in [0.0, 1.0]

        self.transform = transforms.Compose(transform_list)

        # Observation space definition
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.channels, img_size[0], img_size[1]),
            dtype=np.float32,
        )

    def _get_pixel_obs(self) -> torch.Tensor:
        """Render RGB frame and apply transformations."""
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "Environment render() returned None. Ensure the environment is created with render_mode='rgb_array'."
            )
        tensor_img = self.transform(frame)
        return tensor_img

    def reset(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        _, info = self.env.reset(**kwargs)
        obs = self._get_pixel_obs()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_pixel_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info


def make_pixel_env(
    env_id: str = "LunarLander-v3",
    continuous: bool = True,
    img_size: Tuple[int, int] = (64, 64),
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Factory function to instantiate and wrap continuous control environments with pixel observations.

    Handles deprecated environment IDs (e.g., LunarLander-v2 -> LunarLander-v3).
    """
    # Map deprecated names
    if "LunarLanderContinuous" in env_id or "LunarLander-v2" in env_id:
        env_id = "LunarLander-v3"
        continuous = True
    elif "CarRacing-v2" in env_id:
        env_id = "CarRacing-v3"

    kwargs = {"render_mode": "rgb_array"}
    if "LunarLander" in env_id:
        kwargs["continuous"] = continuous

    raw_env = gym.make(env_id, **kwargs)
    if seed is not None:
        raw_env.reset(seed=seed)
        raw_env.action_space.seed(seed)

    wrapped_env = GymPixelWrapper(raw_env, img_size=img_size)
    return wrapped_env
