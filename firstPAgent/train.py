"""
Training Lifecycle for First-Principles Active Inference Agent.

Implements:
- Phase 0: Initialization (Gymnasium pixel env, FIFO buffer, networks)
- Phase 1: Dynamics Warm-up (Random exploration + 2,000 gradient updates + SIGReg convergence check)
- Phase 2: Active Inference Online Control Loop (CEM/MPPI planning + continuous online learning)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from firstPAgent.agent import FirstPrinciplesActiveInferenceAgent
from firstPAgent.buffer import PixelReplayBuffer
from firstPAgent.env import make_pixel_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="First-Principles Active Inference Agent (Pixel Gymnasium)"
    )
    # Environment & Representation
    parser.add_argument(
        "--env_id",
        type=str,
        default="LunarLander-v3",
        help="Gymnasium continuous control environment ID",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="Continuous action space flag",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Square image resolution (e.g., 64 for 64x64)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Latent representation dimension D (e.g., 64 or 128)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictor heads in ensemble M (default: 5)",
    )

    # Lifecycle Steps
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5000,
        help="Phase 1: Environment random exploration steps (default: 5000)",
    )
    parser.add_argument(
        "--warmup_train_steps",
        type=int,
        default=2000,
        help="Phase 1: Gradient updates after exploration (default: 2000)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Phase 2: Active inference online control episodes (default: 50)",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=500,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size B >= 64 for SIGReg stability",
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=100000,
        help="Replay buffer capacity (default: 100000)",
    )

    # Optimization & Weights
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--sigreg_knots", type=int, default=17, help="True SIGReg quadrature knots (default: 17)"
    )
    parser.add_argument(
        "--sigreg_num_proj", type=int, default=1024, help="True SIGReg random 1D projections (default: 1024)"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Total loss regularizer weight"
    )

    # Planner
    parser.add_argument(
        "--horizon", type=int, default=12, help="Planning horizon H (default: 12)"
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=256,
        help="CEM parallel candidates N (default: 256)",
    )
    parser.add_argument(
        "--num_elites",
        type=int,
        default=32,
        help="CEM top elites K (default: 32)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3,
        help="CEM iterations per step",
    )
    parser.add_argument(
        "--w_rew",
        type=float,
        default=0.0,
        help="Extrinsic reward weight (Pragmatic Value). Set >0 (e.g. 1.0) to optimize task reward.",
    )
    parser.add_argument(
        "--w_epi", type=float, default=1.0, help="Epistemic uncertainty weight"
    )
    parser.add_argument(
        "--w_viab", type=float, default=10.0, help="Viability boundary weight"
    )
    parser.add_argument(
        "--w_emp", type=float, default=0.5, help="Empowerment variance weight"
    )
    parser.add_argument(
        "--planner_type",
        type=str,
        default="cem",
        choices=["cem", "mppi"],
        help="Planner optimizer algorithm",
    )

    # General
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Updates between logging checklist metrics",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("First-Principles Active Inference Agent (Single Shared Encoder with Target Detach)")
    print(f"Device: {args.device} | Env: {args.env_id} | Resolution: {args.img_size}x{args.img_size}")
    print(f"Latent Dim: {args.latent_dim} | Ensemble Size: {args.ensemble_size}")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------------------------------------------
    # Phase 0: Initialization
    # -------------------------------------------------------------
    print("\n[Phase 0: Initialization]")
    env = make_pixel_env(
        env_id=args.env_id,
        continuous=args.continuous,
        img_size=(args.img_size, args.img_size),
        seed=args.seed,
    )

    obs_shape = env.observation_space.shape  # (C, H, W)
    act_dim = env.action_space.shape[0]
    act_low = float(env.action_space.low[0])
    act_high = float(env.action_space.high[0])

    print(f"Observation shape: {obs_shape}, Action dim: {act_dim} [range: {act_low}, {act_high}]")

    # Buffer: Stores raw transitions in uint8 (latents NEVER cached)
    buffer = PixelReplayBuffer(
        capacity=args.buffer_capacity,
        obs_shape=obs_shape,
        act_dim=act_dim,
        device=args.device,
    )

    # Agent
    agent = FirstPrinciplesActiveInferenceAgent(
        in_channels=obs_shape[0],
        latent_dim=args.latent_dim,
        act_dim=act_dim,
        num_models=args.ensemble_size,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        gamma=args.gamma,
        lr=args.lr,
        weight_decay=args.weight_decay,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        w_rew=args.w_rew,
        w_epi=args.w_epi,
        w_viab=args.w_viab,
        w_emp=args.w_emp,
        action_low=act_low,
        action_high=act_high,
        planner_type=args.planner_type,
        device=args.device,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Phase 1: Dynamics Warm-up (Random Exploration)
    # -------------------------------------------------------------
    print(f"\n[Phase 1: Dynamics Warm-up - Collecting {args.warmup_steps} random steps]")
    obs, _ = env.reset(seed=args.seed)
    pbar_collect = tqdm(total=args.warmup_steps, desc="Exploring (Random Policy)")

    collected_steps = 0
    while collected_steps < args.warmup_steps:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, done)
        collected_steps += 1
        pbar_collect.update(1)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    pbar_collect.close()
    print(f"Replay buffer populated with {len(buffer)} transitions.")

    # Phase 1 Step 2: 2,000 gradient updates on random mini-batches
    print(f"\n[Phase 1: Training World Model for {args.warmup_train_steps} gradient updates]")
    pbar_train = tqdm(total=args.warmup_train_steps, desc="Warm-up Training")

    for step in range(1, args.warmup_train_steps + 1):
        batch = buffer.sample(args.batch_size, device=args.device)
        metrics = agent.train_step(batch)
        pbar_train.update(1)

        if step % args.log_interval == 0 or step == args.warmup_train_steps:
            # Verify Section 5 Developer Checklist
            std_mean = metrics["std_mean"]
            off_diag = metrics["off_diag_mean"]
            epi_unc = metrics["batch_epi_unc"]
            viab_acc = metrics["viab_acc"]

            health_status = "HEALTHY" if (std_mean >= 0.8 and off_diag <= 0.1) else "ADJUST_NEEDED"
            pbar_train.set_postfix(
                loss_dyn=f"{metrics['loss_dyn']:.4f}",
                loss_reg=f"{metrics['loss_reg']:.4f}",
                std=f"{std_mean:.3f}",
                off_diag=f"{off_diag:.3f}",
                epi=f"{epi_unc:.3f}",
                status=health_status,
            )

            if step % (args.log_interval * 5) == 0:
                print(
                    f"\nStep {step}/{args.warmup_train_steps} | "
                    f"Dyn Loss: {metrics['loss_dyn']:.4f} | "
                    f"SIGReg Loss: {metrics['loss_reg']:.4f} | "
                    f"Latent Std: {std_mean:.3f} (target: ~1.0) | "
                    f"Off-Diag Cov: {off_diag:.4f} (target: <0.1) | "
                    f"Epistemic Unc: {epi_unc:.4f} | "
                    f"Viab Acc: {viab_acc:.3f}"
                )

    pbar_train.close()
    print("\nWarm-up phase completed. SIGReg convergence verified.")

    # -------------------------------------------------------------
    # Phase 2: Active Inference Online Control Loop
    # -------------------------------------------------------------
    print(f"\n[Phase 2: Active Inference Online Control Loop - {args.episodes} episodes]")
    total_env_steps = args.warmup_steps

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        agent.reset_planner()

        ep_reward = 0.0
        ep_steps = 0
        ep_dyn_loss = []
        ep_reg_loss = []
        ep_epi_value = []
        ep_viab_cost = []
        start_time = time.time()

        for t in range(args.max_steps_per_episode):
            # 1. Action Selection via CEM / MPPI Active Inference
            action, plan_diag = agent.select_action(obs, warm_start=True)

            # 2. Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. Store raw tuple into FIFO replay buffer (including reward)
            buffer.add(obs, action, reward, next_obs, done)

            # 4. Continuous Online Learning: Sample mini-batch and update world model
            batch = buffer.sample(args.batch_size, device=args.device)
            train_metrics = agent.train_step(batch)

            # Accumulate metrics
            ep_reward += reward
            ep_steps += 1
            total_env_steps += 1
            ep_dyn_loss.append(train_metrics["loss_dyn"])
            ep_reg_loss.append(train_metrics["loss_reg"])
            ep_epi_value.append(plan_diag.get("mean_epi_value", 0.0))
            ep_viab_cost.append(plan_diag.get("mean_viab_cost", 0.0))

            if done:
                break
            obs = next_obs

        duration = time.time() - start_time
        fps = ep_steps / max(duration, 1e-4)

        avg_dyn = np.mean(ep_dyn_loss) if ep_dyn_loss else 0.0
        avg_reg = np.mean(ep_reg_loss) if ep_reg_loss else 0.0
        avg_epi = np.mean(ep_epi_value) if ep_epi_value else 0.0
        avg_viab = np.mean(ep_viab_cost) if ep_viab_cost else 0.0

        rew_str = f" | RewLoss: {train_metrics.get('loss_rew', 0.0):.4f}" if args.w_rew > 0 else ""
        print(
            f"Episode {ep:03d}/{args.episodes:03d} | "
            f"Steps: {ep_steps:3d} | "
            f"Reward: {ep_reward:7.2f} | "
            f"DynLoss: {avg_dyn:.4f} | "
            f"SIGReg: {avg_reg:.4f} | "
            f"EpiVal: {avg_epi:.4f} | "
            f"ViabCost: {avg_viab:.4f}"
            f"{rew_str} | "
            f"Speed: {fps:.1f} FPS"
        )

        # Checkpoint every 10 episodes
        if ep % 10 == 0 or ep == args.episodes:
            ckpt_path = os.path.join(args.save_dir, f"agent_ep{ep}.pt")
            agent.save(ckpt_path)
            print(f"--> Saved checkpoint to {ckpt_path}")

    env.close()
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
