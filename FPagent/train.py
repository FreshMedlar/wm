"""
Training Lifecycle for First-Principles Active Inference Agent.

Implements Pure Offline World Model Training (Option 1):
- Phase 0: Initialization (Gymnasium pixel env, FIFO buffer, networks)
- Phase 1: High-Speed Data Collection (Random exploration at >500 FPS, no planning)
- Phase 2: High-Throughput Offline Training (Pure GPU batched SGD on buffer, zero planning overhead)
- Phase 3: Active Inference Policy Evaluation (CEM/MPPI trajectory optimization only at test time)

Legacy online interleaved control loop is also available via the --online flag.
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

from FPagent.agent import FirstPrinciplesActiveInferenceAgent
from FPagent.buffer import PixelReplayBuffer
from FPagent.env import make_pixel_env
from FPagent.evaluate import evaluate


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

    # Offline Training Lifecycle (Option 1)
    parser.add_argument(
        "--collect_steps",
        "--warmup_steps",
        dest="collect_steps",
        type=int,
        default=10000,
        help="Phase 1: Environment random exploration steps to collect (default: 10000)",
    )
    parser.add_argument(
        "--train_steps",
        "--warmup_train_steps",
        dest="train_steps",
        type=int,
        default=5000,
        help="Phase 2: Offline gradient updates on the collected replay buffer (default: 5000)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=3,
        help="Phase 3: Number of evaluation episodes with CEM/MPPI planner (default: 3)",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="Interval of train steps to run mid-training evaluation (0 = eval only at end)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Gradient updates between checkpoint saves (default: 1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size B >= 64 for SIGReg stability (default: 64)",
    )
    parser.add_argument(
        "--buffer_capacity",
        type=int,
        default=100000,
        help="Replay buffer capacity (default: 100000)",
    )
    parser.add_argument(
        "--negative_sample_ratio",
        type=float,
        default=0.2,
        help="Fraction of mini-batch transitions prioritized from salient negative/terminal events (default: 0.2)",
    )
    parser.add_argument(
        "--negative_threshold",
        type=float,
        default=-10.0,
        help="Reward threshold to consider a transition a salient penalty (default: -10.0)",
    )

    # Optional Legacy Online Control Loop
    parser.add_argument(
        "--online",
        action="store_true",
        default=False,
        help="Run legacy per-step online active inference control loop instead of pure offline training",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes for online control loop if --online is enabled (default: 50)",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=500,
        help="Max steps per episode during planning rollouts",
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

    # Planner (Used during Evaluation or when --online is active)
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
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to saved model checkpoint to resume training from",
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
    print("First-Principles Active Inference Agent")
    mode_str = "Legacy Online Loop" if args.online else "Pure Offline Training (High Throughput)"
    print(f"Mode: {mode_str}")
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

    if args.resume_checkpoint is not None:
        print(f"\n--> Resuming weights from checkpoint: {args.resume_checkpoint}")
        agent.load(args.resume_checkpoint)

    # -------------------------------------------------------------
    # Option 1: Pure Offline Training Workflow (Default)
    # -------------------------------------------------------------
    if not args.online:
        # Phase 1: High-Speed Data Collection (Zero Planning)
        skip_collection = (args.resume_checkpoint is not None and args.collect_steps == 0)
        if skip_collection:
            print("\n[Phase 1: Skipped because --resume_checkpoint is provided with --collect_steps 0]")
        else:
            print(f"\n[Phase 1: High-Speed Data Collection - Collecting {args.collect_steps} transitions]")
            obs, _ = env.reset(seed=args.seed)
            pbar_collect = tqdm(total=args.collect_steps, desc="Collecting (Fast Random)")
            collect_start = time.time()

            collected_steps = 0
            episodes_collected = 0
            while collected_steps < args.collect_steps:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.add(obs, action, reward, next_obs, done)
                collected_steps += 1
                pbar_collect.update(1)

                if done:
                    episodes_collected += 1
                    obs, _ = env.reset()
                else:
                    obs = next_obs

            pbar_collect.close()
            collect_dur = max(time.time() - collect_start, 1e-4)
            collect_fps = collected_steps / collect_dur
            print(
                f"Data collection completed: {len(buffer)} transitions ({episodes_collected} episodes) "
                f"in {collect_dur:.2f}s ({collect_fps:.1f} FPS)."
            )

        # Phase 2: High-Throughput Offline World Model Training
        print(f"\n[Phase 2: High-Throughput Offline Training - {args.train_steps} gradient updates]")
        pbar_train = tqdm(total=args.train_steps, desc="Offline Training")
        train_start = time.time()

        for step in range(1, args.train_steps + 1):
            batch = buffer.sample(
                args.batch_size,
                device=args.device,
                negative_ratio=args.negative_sample_ratio,
                neg_threshold=args.negative_threshold,
            )
            metrics = agent.train_step(batch)
            pbar_train.update(1)

            if step % args.log_interval == 0 or step == args.train_steps:
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
                    elapsed = max(time.time() - train_start, 1e-4)
                    ups = step / elapsed
                    print(
                        f"\nStep {step:05d}/{args.train_steps:05d} | "
                        f"Dyn Loss: {metrics['loss_dyn']:.4f} | "
                        f"SIGReg Loss: {metrics['loss_reg']:.4f} | "
                        f"Latent Std: {std_mean:.3f} | "
                        f"Off-Diag Cov: {off_diag:.4f} | "
                        f"Epistemic Unc: {epi_unc:.4f} | "
                        f"Viab Acc: {viab_acc:.3f} | "
                        f"Speed: {ups:.1f} updates/s"
                    )

            if step % args.save_interval == 0 or step == args.train_steps:
                ckpt_path = os.path.join(args.save_dir, f"agent_step{step}.pt")
                agent.save(ckpt_path)
                agent.save(os.path.join(args.save_dir, "agent_latest.pt"))

            if args.eval_interval > 0 and step % args.eval_interval == 0 and step < args.train_steps:
                print(f"\n--> Running mid-training evaluation at step {step}...")
                evaluate(
                    agent,
                    env,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_steps_per_episode,
                )

        pbar_train.close()
        train_dur = max(time.time() - train_start, 1e-4)
        print(
            f"\nOffline training finished: {args.train_steps} updates in {train_dur:.2f}s "
            f"({args.train_steps / train_dur:.1f} updates/s)."
        )

        # Phase 3: Active Inference Benchmark Evaluation
        if args.eval_episodes > 0:
            print(f"\n[Phase 3: Active Inference Benchmark Evaluation - {args.eval_episodes} episodes]")
            eval_results = evaluate(
                agent,
                env,
                num_episodes=args.eval_episodes,
                max_steps=args.max_steps_per_episode,
            )
            print("\n" + "=" * 60)
            print("Evaluation Summary:")
            for k, v in eval_results.items():
                print(f"  {k:28s}: {v:.4f}")
            print("=" * 60)

    # -------------------------------------------------------------
    # Legacy Option: Online Control Loop
    # -------------------------------------------------------------
    else:
        print(f"\n[Phase 1: Warm-up Collection - {args.collect_steps} steps]")
        obs, _ = env.reset(seed=args.seed)
        pbar_collect = tqdm(total=args.collect_steps, desc="Exploring (Random)")
        for _ in range(args.collect_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, done)
            pbar_collect.update(1)
            obs = env.reset()[0] if done else next_obs
        pbar_collect.close()

        print(f"\n[Phase 2: Active Inference Online Control Loop - {args.episodes} episodes]")
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            agent.reset_planner()
            ep_reward = 0.0
            ep_steps = 0
            start_time = time.time()

            for t in range(args.max_steps_per_episode):
                action, plan_diag = agent.select_action(obs, warm_start=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.add(obs, action, reward, next_obs, done)
                batch = buffer.sample(args.batch_size, device=args.device)
                train_metrics = agent.train_step(batch)

                ep_reward += reward
                ep_steps += 1
                if done:
                    break
                obs = next_obs

            duration = time.time() - start_time
            fps = ep_steps / max(duration, 1e-4)
            print(
                f"Episode {ep:03d}/{args.episodes:03d} | "
                f"Steps: {ep_steps:3d} | "
                f"Reward: {ep_reward:7.2f} | "
                f"DynLoss: {train_metrics['loss_dyn']:.4f} | "
                f"SIGReg: {train_metrics['loss_reg']:.4f} | "
                f"Speed: {fps:.1f} FPS"
            )
            if ep % 10 == 0 or ep == args.episodes:
                ckpt_path = os.path.join(args.save_dir, f"agent_ep{ep}.pt")
                agent.save(ckpt_path)

    env.close()
    print("\nLifecycle completed successfully!")


if __name__ == "__main__":
    main()
