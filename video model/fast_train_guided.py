"""Guided Latent Flow-Matching Evolution Training with Trained JEPA Upsampler.

Pipeline:
1. Loads precomputed cache (500 clips).
2. Loads pretrained frozen JEPAUpsampler to compute high-res guidance:
       z_guide = JEPAUpsampler(z_{t-1}, jepa_tokens_t)
3. LatentEvolutionDiT trains with both:
   - Flow matching vector field loss: ||v_pred - (z_t - z_{t-1})||^2
   - Direct 1-step terminal guidance alignment loss: ||z_pred_1 - z_t||_1
4. Evaluates autoregressive rollouts and logs comparison GIFs to Weights & Biases.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/medlar/Projects")

from diffusers import AutoencoderKL
from model import LatentEvolutionDiT
from jepa_upsampler import JEPAUpsampler
from cache_features import CachedLatentDataset
from flow_solver import solve_flow_bridge


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def save_video_gif(frames_tensor: torch.Tensor, output_path: str, fps: int = 8):
    """Saves [T, 3, H, W] tensor in range [-1, 1] as an animated GIF."""
    frames_clamped = ((frames_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).to(torch.uint8)
    pil_frames = []
    for t in range(frames_clamped.shape[0]):
        img_np = frames_clamped[t].permute(1, 2, 0).cpu().numpy()
        pil_frames.append(Image.fromarray(img_np))
    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train Latent Evolution DiT Guided by Upsampler")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="DiT hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="DiT depth")
    parser.add_argument("--num-heads", type=int, default=4, help="DiT attention heads")
    parser.add_argument("--guide-weight", type=float, default=0.2, help="Terminal step L1 loss weight")
    parser.add_argument("--eval-every", type=int, default=10, help="Save rollout and log gif every N epochs")
    parser.add_argument("--upsampler-ckpt", type=str, default=str(SCRIPT_DIR / "outputs_upsampler/upsampler_epoch_50.pt"), help="Upsampler checkpoint path")
    parser.add_argument("--cache-dir", type=str, default=str(SCRIPT_DIR / "latent_cache"), help="Cache directory")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR / "outputs_guided"), help="Output directory")
    parser.add_argument("--project", type=str, default="video-evolution-model", help="W&B project name")
    parser.add_argument("--run-name", type=str, default="guided-dit-evolution-run-1", help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "hidden_size": args.hidden_size,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "guide_weight": args.guide_weight,
                "device": str(device),
                "model": "GuidedLatentEvolutionDiT",
            },
        )
        print(f"W&B initialized! Run URL: {wandb.run.url if wandb.run else 'N/A'}")

    # 1. Load Pretrained Frozen Upsampler
    print(f"Loading Frozen JEPAUpsampler from {args.upsampler_ckpt}...")
    upsampler = JEPAUpsampler(latent_channels=4, jepa_dim=1024, hidden_dim=128, target_size=32).to(device)
    if os.path.exists(args.upsampler_ckpt):
        upsampler.load_state_dict(torch.load(args.upsampler_ckpt, map_location=device))
        print("Upsampler weights loaded successfully!")
    else:
        # Fallback to latest available checkpoint if epoch 50 is not found
        fallback_ckpts = sorted(list((SCRIPT_DIR / "outputs_upsampler").glob("upsampler_epoch_*.pt")))
        if fallback_ckpts:
            latest = fallback_ckpts[-1]
            print(f"Warning: {args.upsampler_ckpt} not found. Loading fallback: {latest}")
            upsampler.load_state_dict(torch.load(latest, map_location=device))
        else:
            raise FileNotFoundError(f"No upsampler checkpoints found in {SCRIPT_DIR / 'outputs_upsampler'}")
    upsampler.eval()
    for param in upsampler.parameters():
        param.requires_grad = False

    # 2. Load Cached Dataset
    dataset = CachedLatentDataset(cache_dir=args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} cached video sequences ({len(dataloader)} batches/epoch).")

    # 3. Instantiate Trainable DiT
    model = LatentEvolutionDiT(
        latent_channels=4,
        context_frames=2,
        latent_size=32,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        jepa_dim=1024,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in LatentEvolutionDiT: {trainable_params:,}")
    if use_wandb:
        wandb.config.update({"trainable_params": trainable_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    vae = None
    global_step = 0
    print("Beginning Guided Flow-Matching Training...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_flow = 0.0
        total_guide = 0.0

        for step, batch in enumerate(dataloader):
            global_step += 1
            latents = batch["latents"].to(device)       # [B, T=16, 4, 32, 32]
            jepa_tokens = batch["jepa_tokens"].to(device) # [B, T=16, 196, 1024]
            B, T, C, H, W = latents.shape

            # Random transition step t in [2, T-1]
            t = torch.randint(2, T, (1,)).item()

            z_curr = latents[:, t - 1]   # z_{t-1}
            z_target = latents[:, t]     # z_t
            z_prev = latents[:, t - 2]   # z_{t-2}
            jepa_target_tokens = jepa_tokens[:, t]  # [B, 196, 1024]

            # 1. Compute High-Resolution Guidance using the frozen upsampler
            with torch.no_grad():
                z_guide = upsampler(z_curr, jepa_target_tokens)  # [B, 4, 32, 32]

            # Context latents: [z_{t-1}, z_{t-2}] with slight contractive perturbation
            noise_pert = torch.randn_like(z_curr) * 0.02
            z_ctx = torch.cat([z_curr + noise_pert, z_prev], dim=1)

            # Continuous optimal transport bridge
            s = torch.rand(B, device=device)
            s_expanded = s.view(B, 1, 1, 1)
            z_s = (1.0 - s_expanded) * z_curr + s_expanded * z_target
            u_target = z_target - z_curr

            # Predict velocity
            v_pred = model(z_s, s, z_ctx, z_guide, jepa_target_tokens)
            flow_loss = F.mse_loss(v_pred, u_target)

            # Terminal guidance step (predicts next latent directly at s=0)
            # v_init = model(z_curr, s=0, ...) -> z_pred_1 = z_curr + v_init
            s_zero = torch.zeros(B, device=device)
            v_zero = model(z_curr, s_zero, z_ctx, z_guide, jepa_target_tokens)
            z_pred_1 = z_curr + v_zero
            guide_l1_loss = F.l1_loss(z_pred_1, z_target)

            loss = flow_loss + args.guide_weight * guide_l1_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_flow += flow_loss.item()
            total_guide += guide_l1_loss.item()

            if use_wandb and global_step % 10 == 0:
                wandb.log({
                    "step_loss": loss.item(),
                    "step_flow_loss": flow_loss.item(),
                    "step_guide_l1": guide_l1_loss.item(),
                    "global_step": global_step,
                })

        scheduler.step()
        num_batches = max(1, len(dataloader))
        avg_loss = total_loss / num_batches
        avg_flow = total_flow / num_batches
        avg_guide = total_guide / num_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Total Loss: {avg_loss:.5f} (Flow: {avg_flow:.5f}, Guide L1: {avg_guide:.5f}) | LR: {current_lr:.6f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_loss": avg_loss,
                "epoch_flow_loss": avg_flow,
                "epoch_guide_loss": avg_guide,
                "learning_rate": current_lr,
            })

        # Periodic rollout evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"guided_dit_epoch_{epoch:02d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

            if vae is None:
                print("Loading VAE for visual evaluation...")
                vae = AutoencoderKL.from_pretrained("/home/medlar/Projects/sd-vae-ft-mse", local_files_only=True).to(device).eval()

            gif_path = evaluate_guided_rollout(
                model=model,
                upsampler=upsampler,
                vae=vae,
                sample_dict=dataset[0],
                device=device,
                epoch=epoch,
                out_dir=args.out_dir,
            )

            if use_wandb and gif_path and os.path.exists(gif_path):
                wandb.log({
                    "rollout_comparison": wandb.Video(gif_path, format="gif", fps=6, caption=f"Epoch {epoch} (Left: GT, Right: Guided DiT)"),
                    "epoch": epoch,
                })

    if use_wandb:
        wandb.finish()
    print("\nTraining completed successfully!")


@torch.no_grad()
def evaluate_guided_rollout(model, upsampler, vae, sample_dict, device, epoch, out_dir):
    """Rolls out 16 frames autoregressively using guided DiT and decodes via VAE."""
    model.eval()
    upsampler.eval()

    latents_gt = sample_dict["latents"].unsqueeze(0).to(device)       # [1, 16, 4, 32, 32]
    jepa_tokens = sample_dict["jepa_tokens"].unsqueeze(0).to(device)  # [1, 16, 196, 1024]
    T = latents_gt.shape[1]

    # Initial conditions: t=0, t=1
    rolled_latents = [latents_gt[:, 0], latents_gt[:, 1]]

    for t in range(2, T):
        z_curr = rolled_latents[-1]
        z_prev = rolled_latents[-2]
        z_ctx = torch.cat([z_curr, z_prev], dim=1)
        jepa_target = jepa_tokens[:, t]

        # 1. Compute high-res guidance
        z_guide = upsampler(z_curr, jepa_target)

        # 2. Solve ODE bridge with guidance (2 Heun steps)
        z_next = solve_flow_bridge(
            model=model,
            z_init=z_curr,
            z_ctx=z_ctx,
            z_guide=z_guide,
            jepa_tokens=jepa_target,
            steps=2,
            method="heun",
        )
        rolled_latents.append(z_next)

    rolled_latents_tensor = torch.stack(rolled_latents, dim=1)  # [1, 16, 4, 32, 32]
    vae_scaling_factor = vae.config.scaling_factor or 0.18215

    # Decode GT and Generated
    gt_decoded = vae.decode(latents_gt.squeeze(0) / vae_scaling_factor).sample
    gen_decoded = vae.decode(rolled_latents_tensor.squeeze(0) / vae_scaling_factor).sample

    comparison_frames = []
    for t in range(T):
        combined = torch.cat([gt_decoded[t], gen_decoded[t]], dim=2)  # [3, 256, 512]
        comparison_frames.append(combined)
    comparison_tensor = torch.stack(comparison_frames, dim=0)

    gif_path = os.path.join(out_dir, f"guided_rollout_epoch_{epoch:02d}.gif")
    save_video_gif(comparison_tensor, gif_path, fps=6)
    print(f"Side-by-side rollout GIF saved to: {gif_path}")
    return gif_path


if __name__ == "__main__":
    main()
