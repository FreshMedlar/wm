from transformers.generation.continuous_batching import scheduler
import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from models.module import SIGReg

class LunarLanderImageWrapper(gym.ObservationWrapper):
    """
    Gymnasium ObservationWrapper for LunarLander that produces image observations
    instead of the default 8-dimensional state vector.

    Outputs frames as torch.Tensor formatted as (C, H, W) normalized to [0, 1].
    """

    def __init__(self, env: gym.Env, img_size=(84, 84)):
        super().__init__(env)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),  # Converts uint8 (H, W, C) -> float32 (C, H, W) in [0, 1]
        ])

        # Define the new image observation space: (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, img_size[0], img_size[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        # Render RGB frame: (H, W, C)
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Environment render() returned None. Ensure render_mode='rgb_array'.")

        tensor_img = self.transform(frame)
        return tensor_img


def make_lunar_lander_image_env(img_size=(84, 84), continuous=False):
    """
    Factory function to create a LunarLander environment with image observations.
    """
    env_id = "LunarLander-v3"
    raw_env = gym.make(env_id, continuous=continuous, render_mode="rgb_array")
    env = LunarLanderImageWrapper(raw_env, img_size=img_size)
    return env

def train(
    model,
    env,
    episodes,
    max_steps,
    lr,
    weight_decay,
    ctx_len,
    n_preds, # target pred offset
    lambd, # for sigreg reg weight
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)

    # 1
    sigreg = SIGReg().to(device)
    # 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=episodes) # previously episodes was num_epochs, may cause problems

    for episode in range(episodes):
        model.train()
        total_train_loss = 0.0
        total_pred_loss = 0.0
        total_sigreg_loss = 0.0


        for step in max_steps:

            
            obs, reward, terminated, truncated, info = env.step(action)
    return

if __name__ == "__main__":
    print("Testing LunarLander image environment...")
    env = make_lunar_lander_image_env(img_size=(84, 84))

    obs, info = env.reset(seed=42)
    print(f"Reset observation type: {type(obs)}, shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Pixel value range: min={obs.min().item():.3f}, max={obs.max().item():.3f}")

    # Take random steps to verify stepping
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: obs shape={obs.shape}, reward={reward:.3f}")
        if terminated or truncated:
            break

    env.close()
    print("Environment test passed successfully!")
