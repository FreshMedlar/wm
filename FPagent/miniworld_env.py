"""
MiniWorld Continuous Navigation Environment Wrapper with Contextual Language Instructions.

Wraps MiniWorld (e.g. MiniWorld-PickupObjects-v0 or MiniWorld-Hallway-v0) to:
1. Provide continuous control (forward/backward velocity, angular steering rate).
2. Generate standardized (C, H, W) normalized visual observations.
3. Emit natural language instructions per episode (e.g., "reach the red box", "avoid grey zone").
"""

from typing import List, Optional, Tuple, Union
import gymnasium as gym
import miniworld  # Registers MiniWorld environments with Gymnasium
import numpy as np
import torch
from torchvision import transforms


class MiniWorldLanguageWrapper(gym.Wrapper):
    """
    Continuous Action & Language Instruction Wrapper for 3D MiniWorld environments.

    Action Space:
      Continuous Box(2,) in [-1.0, 1.0]:
        action[0]: forward / backward translation rate
        action[1]: angular turn rate (left / right)
    """

    def __init__(
        self,
        env_id: str = "MiniWorld-PickupObjects-v0",
        img_size: Tuple[int, int] = (64, 64),
        max_episode_steps: int = 150,
    ):
        raw_env = gym.make(env_id, max_episode_steps=max_episode_steps)
        super().__init__(raw_env)

        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),  # (H, W, C) uint8 -> (C, H, W) float in [0, 1]
        ])

        # Continuous action space: [translation, rotation]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: (3, H, W)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, img_size[0], img_size[1]), dtype=np.float32
        )

        self.current_instruction: str = ""
        self.target_color: Optional[str] = None
        self.hazard_color: Optional[str] = None

    def _format_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        """Convert MiniWorld rendered frame (H, W, C) to (C, H, W) torch tensor."""
        return self.transform(obs_np)

    def _sample_instruction(self) -> str:
        """Inspect room entities and sample a semantic task description."""
        unwrapped = self.env.unwrapped
        colors_present = []
        if hasattr(unwrapped, "entities"):
            for ent in unwrapped.entities:
                if hasattr(ent, "color") and ent.color:
                    colors_present.append(str(ent.color))

        if not colors_present:
            colors_present = ["red", "green", "blue", "grey"]

        # Pick random target from objects present
        self.target_color = str(np.random.choice(colors_present))
        other_colors = [c for c in colors_present if c != self.target_color]
        if other_colors and np.random.rand() > 0.5:
            self.hazard_color = str(np.random.choice(other_colors))
            instruction = f"reach the {self.target_color} object, avoid {self.hazard_color}"
        else:
            self.hazard_color = None
            instruction = f"reach the {self.target_color} object"

        return instruction

    def reset(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        raw_obs, info = self.env.reset(**kwargs)
        self.current_instruction = self._sample_instruction()
        info["instruction"] = self.current_instruction

        obs = self._format_obs(raw_obs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, dict]:
        """
        Execute continuous 2D action in MiniWorld.
        Maps continuous [forward, turn] into MiniWorld agent kinematics.
        """
        action = np.clip(action, -1.0, 1.0)
        forward_val = float(action[0])
        turn_val = float(action[1])

        unwrapped = self.env.unwrapped

        # Apply continuous turn rate
        turn_angle = turn_val * 15.0  # degrees
        unwrapped.turn_agent(turn_angle)

        # Apply continuous forward/backward movement
        step_distance = forward_val * (unwrapped.max_forward_step or 0.15)
        unwrapped.move_agent(step_distance, 0)

        # Step time counter & check goal conditions
        unwrapped.step_count += 1
        raw_obs = unwrapped.render_obs()

        # Check task completion and goal reward
        reward = 0.0
        terminated = False

        agent_pos = unwrapped.agent.pos
        for ent in unwrapped.entities:
            if hasattr(ent, "color") and ent.color:
                ent_color = str(ent.color)
                dist = np.linalg.norm(agent_pos - ent.pos)

                if ent_color == self.target_color:
                    # Sparse arrival reward + dense proximity shaping
                    if unwrapped.near(ent) or dist < 1.2:
                        reward += 10.0
                        terminated = True
                    else:
                        reward += max(0.0, (10.0 - dist) * 0.05)

                elif self.hazard_color and ent_color == self.hazard_color:
                    # Penalty for colliding or nearing hazard
                    if unwrapped.near(ent) or dist < 1.2:
                        reward -= 5.0
                        terminated = True

        done = terminated or (unwrapped.step_count >= unwrapped.max_episode_steps)

        obs = self._format_obs(raw_obs)
        info = {
            "instruction": self.current_instruction,
            "target_color": self.target_color,
            "hazard_color": self.hazard_color,
        }

        return obs, float(reward), bool(terminated), bool(done and not terminated), info
