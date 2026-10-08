"""
Record 3 Evaluation Episodes as Animated GIFs using saved model checkpoint.
Captures high-resolution environment frames with diagnostic HUD overlay.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from FPagent.agent import FirstPrinciplesActiveInferenceAgent
from FPagent.env import make_pixel_env


def record_eval_episodes(
    checkpoint_path: str = "./checkpoints/lunar_offline/agent_latest.pt",
    env_id: str = "LunarLander-v3",
    output_dirs = None,
    num_episodes: int = 3,
    max_steps: int = 250,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if output_dirs is None:
        output_dirs = ["./eval_gifs"]

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    env = make_pixel_env(env_id=env_id, continuous=True, img_size=(64, 64))
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Initialize agent matching exact checkpoint configuration
    agent = FirstPrinciplesActiveInferenceAgent(
        in_channels=obs_shape[0],
        latent_dim=64,
        act_dim=act_dim,
        num_models=5,
        horizon=12,
        num_candidates=256,
        num_elites=32,
        num_iterations=3,
        w_rew=0.0,
        w_epi=1.0,
        w_viab=10.0,
        w_emp=0.5,
        planner_type="cem",
        device=device,
    )
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    episode_summaries = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=100 + ep)
        agent.reset_planner()
        frames = []
        ep_reward = 0.0

        print(f"\nRecording Episode {ep}/{num_episodes}...")
        for t in range(max_steps):
            # Capture high-res rendered RGB frame (400x600)
            raw_frame = env.env.render()
            if raw_frame is not None:
                img = Image.fromarray(raw_frame)
                # Overlay diagnostic HUD
                draw = ImageDraw.Draw(img)
                hud_text = f"Ep {ep} | Step {t:3d} | CumReward: {ep_reward:7.1f}"
                draw.rectangle([(5, 5), (320, 28)], fill=(0, 0, 0, 180))
                draw.text((10, 8), hud_text, fill=(255, 255, 255))
                frames.append(img)

            action, plan_diag = agent.select_action(obs, warm_start=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            obs = next_obs
            if done:
                # Capture terminal frame
                raw_frame = env.env.render()
                if raw_frame is not None:
                    img = Image.fromarray(raw_frame)
                    draw = ImageDraw.Draw(img)
                    status_str = "CRASH / TERMINATED" if terminated else "TRUNCATED"
                    hud_text = f"Ep {ep} | Final Step {t+1} | Total: {ep_reward:7.1f} | {status_str}"
                    draw.rectangle([(5, 5), (420, 28)], fill=(180, 0, 0, 200))
                    draw.text((10, 8), hud_text, fill=(255, 255, 255))
                    frames.append(img)
                break

        print(f"Episode {ep} finished: {len(frames)} frames, Total Reward: {ep_reward:.2f}")
        episode_summaries.append((ep, len(frames), ep_reward))

        # Save GIF to all output directories
        gif_filename = f"lunar_eval_ep{ep}.gif"
        for d in output_dirs:
            save_path = os.path.join(d, gif_filename)
            if frames:
                # Subsample slightly if very long, or save at ~20 fps (50ms per frame)
                frames[0].save(
                    save_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=50,  # 20 FPS
                    loop=0,
                    optimize=True,
                )
                file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                print(f"--> Saved: {save_path} ({file_size_mb:.2f} MB)")

    env.close()
    return episode_summaries


if __name__ == "__main__":
    artifact_dir = "/home/medlar/.gemini/antigravity-ide/brain/60aa8a69-e012-4c10-abc0-e8cb3512d1a2"
    workspace_dir = "/home/medlar/Projects/wm/eval_gifs"
    record_eval_episodes(
        checkpoint_path="./checkpoints/lunar_offline/agent_latest.pt",
        output_dirs=[workspace_dir, artifact_dir],
        num_episodes=3,
        max_steps=200,
    )
