"""Training Script for Guided JEPA Latent Upsampler with W&B Logging.

Trains JEPAUpsampler to translate:
    (z_{t-1}, z_t^{jepa}) -> z_t
using:
    - Direct L1 reconstruction loss
    - Spatial gradient / Sobel edge loss (enforces crisp boundaries)
    - Autoregressive rollout evaluation decoded via VAE and logged to W&B
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
from jepa_upsampler import JEPAUpsampler
from cache_features import CachedLatentDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes L1 difference between spatial finite differences (horizontal & vertical edges)."""
    # Horizontal gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Vertical gradients
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    return grad_loss


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
    parser = argparse.ArgumentParser(description="Train Guided JEPA Latent Upsampler")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Upsampler hidden dimension")
    parser.add_argument("--edge-weight", type=float, default=0.5, help="Weight for spatial gradient edge loss")
    parser.add_argument("--eval-every", type=int, default=5, help="Save rollout and log gif every N epochs")
    parser.add_argument("--cache-dir", type=str, default=str(SCRIPT_DIR / "latent_cache"), help="Cache directory")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR / "outputs_upsampler"), help="Output directory")
    parser.add_argument("--project", type=str, default="video-evolution-model", help="W&B project name")
    parser.add_argument("--run-name", type=str, default="jepa-upsampler-run-1", help="W&B run name")
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
                "hidden_dim": args.hidden_dim,
                "edge_weight": args.edge_weight,
                "device": str(device),
                "model": "JEPAUpsampler",
            },
        )
        print(f"W&B initialized! Run URL: {wandb.run.url if wandb.run else 'N/A'}")

    # 1. Load Cached Dataset
    dataset = CachedLatentDataset(cache_dir=args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} cached video sequences ({len(dataloader)} batches/epoch).")

    # 2. Instantiate Model
    model = JEPAUpsampler(
        latent_channels=4,
        jepa_dim=1024,
        hidden_dim=args.hidden_dim,
        target_size=32,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in JEPAUpsampler: {trainable_params:,}")
    if use_wandb:
        wandb.config.update({"trainable_params": trainable_params})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    vae = None
    global_step = 0

    print("Beginning Upsampler Training...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_edge = 0.0

        for step, batch in enumerate(dataloader):
            global_step += 1
            latents = batch["latents"].to(device)         # [B, T=16, 4, 32, 32]
            jepa_tokens = batch["jepa_tokens"].to(device) # [B, T=16, 196, 1024]
            B, T, C, H, W = latents.shape

            # Sample random transition step t in [1, T-1]
            t = torch.randint(1, T, (1,)).item()

            z_prev = latents[:, t - 1]              # [B, 4, 32, 32]
            z_target = latents[:, t]                # [B, 4, 32, 32]
            jepa_target_tokens = jepa_tokens[:, t]  # [B, 196, 1024]

            # Forward pass: direct prediction of z_t
            z_pred = model(z_prev, jepa_target_tokens)

            # Combined L1 + Edge Loss
            l1_loss = F.l1_loss(z_pred, z_target)
            edge_loss = spatial_gradient_loss(z_pred, z_target)
            loss = l1_loss + args.edge_weight * edge_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_edge += edge_loss.item()

            if use_wandb and global_step % 10 == 0:
                wandb.log({
                    "step_loss": loss.item(),
                    "step_l1": l1_loss.item(),
                    "step_edge": edge_loss.item(),
                    "global_step": global_step,
                })

        scheduler.step()
        num_batches = max(1, len(dataloader))
        avg_loss = total_loss / num_batches
        avg_l1 = total_l1 / num_batches
        avg_edge = total_edge / num_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.5f} (L1: {avg_l1:.5f}, Edge: {avg_edge:.5f}) | LR: {current_lr:.6f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_loss": avg_loss,
                "epoch_l1": avg_l1,
                "epoch_edge": avg_edge,
                "learning_rate": current_lr,
            })

        # Periodic rollout evaluation & GIF generation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"upsampler_epoch_{epoch:02d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

            if vae is None:
                print("Loading VAE for visual evaluation...")
                vae = AutoencoderKL.from_pretrained("/home/medlar/Projects/sd-vae-ft-mse", local_files_only=True).to(device).eval()

            gif_path = evaluate_upsampler_rollout(
                model=model,
                vae=vae,
                sample_dict=dataset[0],
                device=device,
                epoch=epoch,
                out_dir=args.out_dir,
            )

            if use_wandb and gif_path and os.path.exists(gif_path):
                wandb.log({
                    "rollout_comparison": wandb.Video(gif_path, format="gif", fps=6, caption=f"Epoch {epoch} (Left: GT, Right: Upsampler)"),
                    "epoch": epoch,
                })

    if use_wandb:
        wandb.finish()
    print("\nTraining completed successfully!")


@torch.no_grad()
def evaluate_upsampler_rollout(model, vae, sample_dict, device, epoch, out_dir):
    """Performs an autoregressive rollout of 16 frames with the upsampler and decodes via VAE."""
    model.eval()

    latents_gt = sample_dict["latents"].unsqueeze(0).to(device)       # [1, 16, 4, 32, 32]
    jepa_tokens = sample_dict["jepa_tokens"].unsqueeze(0).to(device)  # [1, 16, 196, 1024]
    T = latents_gt.shape[1]

    # Initial condition: ground truth t=0
    rolled_latents = [latents_gt[:, 0]]

    # Step-by-step rollout: z_t = Upsampler(z_{t-1}, jepa_t)
    for t in range(1, T):
        z_curr = rolled_latents[-1]
        jepa_t = jepa_tokens[:, t]
        z_next = model(z_curr, jepa_t)
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

    gif_path = os.path.join(out_dir, f"rollout_epoch_{epoch:02d}.gif")
    save_video_gif(comparison_tensor, gif_path, fps=6)
    print(f"Side-by-side rollout GIF saved to: {gif_path}")
    return gif_path


if __name__ == "__main__":
    main()
