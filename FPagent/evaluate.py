"""
Evaluation and Diagnostic Visualization for First-Principles Active Inference Agent.

Runs policy rollouts in the Gymnasium pixel environment, logs expected free energy
components (epistemic uncertainty, viability penalty, empowerment), and reports performance.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import miniworld before torch to avoid Linux GLX driver conflicts
try:
    import miniworld
except ImportError:
    pass

import numpy as np
import torch

from firstPAgent.agent import FirstPrinciplesActiveInferenceAgent
from firstPAgent.env import make_pixel_env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Active Inference Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--env_id", type=str, default="MiniWorld-PickupObjects-v0", help="Gym/MiniWorld environment")
    parser.add_argument("--img_size", type=int, default=64, help="Image resolution")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--horizon", type=int, default=10, help="Planning horizon")
    parser.add_argument("--num_candidates", type=int, default=128, help="Candidates")
    parser.add_argument("--w_rew", type=float, default=0.0, help="Extrinsic reward weight (Pragmatic Value)")
    parser.add_argument("--render_video", action="store_true", help="Record and save rollout animation (GIF)")
    parser.add_argument("--video_path", type=str, default="rollout.gif", help="Path to save animated GIF")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(
    agent: FirstPrinciplesActiveInferenceAgent,
    env,
    lang_encoder=None,
    num_episodes: int = 5,
    max_steps: int = 150,
    render_video: bool = False,
    video_path: str = "rollout.gif",
) -> Dict[str, float]:
    """Runs evaluation rollouts and returns summary statistics, optionally saving GIF animation."""
    from PIL import Image

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    epi_uncertainties: List[float] = []
    viab_penalties: List[float] = []
    empowerments: List[float] = []
    frames: List[Image.Image] = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        context_emb = None
        instruction_str = ""
        if lang_encoder and "instruction" in info:
            instruction_str = info["instruction"]
            context_emb = lang_encoder(instruction_str)

        agent.reset_planner()
        ep_reward = 0.0

        for t in range(max_steps):
            if render_video and ep == 1:
                # Capture frame from obs tensor (3, H, W)
                frame_np = (obs.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                frames.append(Image.fromarray(frame_np))

            action, plan_diag = agent.select_action(obs, context=context_emb, warm_start=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            epi_uncertainties.append(plan_diag.get("mean_epi_value", 0.0))
            viab_penalties.append(plan_diag.get("mean_viab_cost", 0.0))
            empowerments.append(plan_diag.get("mean_empowerment", 0.0))

            obs = next_obs
            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(t + 1)
        instr_print = f" | Prompt: '{instruction_str}'" if instruction_str else ""
        print(
            f"Eval Episode {ep}/{num_episodes} | "
            f"Steps: {t + 1:3d} | "
            f"Reward: {ep_reward:7.2f}"
            f"{instr_print}"
        )

    if render_video and frames:
        # Save as looping animated GIF
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        frames[0].save(
            video_path,
            save_all=True,
            append_images=frames[1:],
            duration=60,  # ~16 FPS
            loop=0,
        )
        print(f"\n--> Saved rollout animation to: {video_path} ({len(frames)} frames)")

    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_epistemic_uncertainty": float(np.mean(epi_uncertainties)),
        "mean_viability_cost": float(np.mean(viab_penalties)),
        "mean_empowerment": float(np.mean(empowerments)),
    }
    return results


def main():
    args = parse_args()
    print("=" * 60)
    print("Evaluating First-Principles Active Inference Agent")
    print(f"Checkpoint: {args.checkpoint} | Env: {args.env_id}")
    print("=" * 60)

    use_language = "miniworld" in args.env_id.lower()
    if use_language:
        from firstPAgent.miniworld_env import MiniWorldLanguageWrapper
        from firstPAgent.language import LanguageContextEncoder
        env = MiniWorldLanguageWrapper(
            env_id=args.env_id,
            img_size=(args.img_size, args.img_size),
            max_episode_steps=args.max_steps,
        )
        context_dim = 64
        lang_encoder = LanguageContextEncoder(context_dim=context_dim, device=args.device)
    else:
        env = make_pixel_env(env_id=args.env_id, img_size=(args.img_size, args.img_size))
        context_dim = 0
        lang_encoder = None

    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = FirstPrinciplesActiveInferenceAgent(
        in_channels=obs_shape[0],
        latent_dim=args.latent_dim,
        act_dim=act_dim,
        context_dim=context_dim,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        w_rew=args.w_rew,
        device=args.device,
    )
    agent.load(args.checkpoint)
    print("Checkpoint loaded successfully. Beginning rollouts...")

    results = evaluate(
        agent,
        env,
        lang_encoder=lang_encoder,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render_video=args.render_video,
        video_path=args.video_path,
    )
    print("\n" + "=" * 60)
    print("Evaluation Summary:")
    for k, v in results.items():
        print(f"  {k:28s}: {v:.4f}")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    main()
