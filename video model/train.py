"""Training & Validation Script for Latent Flow-Matching Evolution Engine.

Features:
- Frozen LeV-JEPA (ImageNet normalized 224x224, 16 frames, tubelet=1, block-causal ViT-L/16)
- Frozen SD-VAE (MSE-fine-tuned, [-1, 1] 256x256 -> 32x32x4 latent space)
- Trainable LatentEvolutionDiT with RoPE Alignment Adapter
- Optimal Transport Flow-Matching objective: ||v_theta - (z_t - z_{t-1})||^2
- Student-Forcing / Latent Perturbation for auto-regressive error correction
- Periodic sample rollout visualizer saving animated comparison GIFs
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

# Ensure imports from current folder and parent folder work
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/medlar/Projects")

from diffusers import AutoencoderKL
from transformers import AutoModel, AutoConfig

from dataset import SyntheticBouncingShapesDataset
from model import LatentEvolutionDiT
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


def main():
    parser = argparse.ArgumentParser(description="Train Latent Evolution DiT")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of toy video sequences")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length in frames")
    parser.add_argument("--eval-every", type=int, default=2, help="Evaluate every N epochs")
    parser.add_argument("--hidden-size", type=int, default=384, help="DiT hidden size")
    parser.add_argument("--depth", type=int, default=4, help="DiT depth")
    parser.add_argument("--num-heads", type=int, default=6, help="DiT attention heads")
    parser.add_argument("--out-dir", type=str, default=str(SCRIPT_DIR / "outputs"), help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Frozen Models
    print("Loading Frozen VAE...")
    vae = AutoencoderKL.from_pretrained("/home/medlar/Projects/sd-vae-ft-mse", local_files_only=True)
    vae = vae.to(device).eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae_scaling_factor = vae.config.scaling_factor or 0.18215

    print("Loading Frozen LeV-JEPA...")
    jepa_config = AutoConfig.from_pretrained("/home/medlar/Projects/LeVJEPA", trust_remote_code=True, local_files_only=True)
    jepa_model = AutoModel.from_pretrained("/home/medlar/Projects/LeVJEPA", config=jepa_config, trust_remote_code=True, local_files_only=True)
    jepa_model = jepa_model.to(device).eval()
    for param in jepa_model.parameters():
        param.requires_grad = False

    # 2. Instantiate Trainable DiT
    print("Instantiating Latent Evolution DiT...")
    model = LatentEvolutionDiT(
        latent_channels=4,
        context_frames=2,
        latent_size=32,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        jepa_dim=jepa_config.embed_dim,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 3. Create Dataset & DataLoader
    dataset = SyntheticBouncingShapesDataset(
        num_samples=args.num_samples,
        seq_length=args.seq_len,
        num_objects=3,
        vae_size=256,
        jepa_size=224,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"Starting training for {args.epochs} epochs ({len(dataloader)} batches/epoch)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            vae_frames = batch["vae_frames"].to(device)   # [B, T=16, 3, 256, 256]
            jepa_frames = batch["jepa_frames"].to(device) # [B, 3, T=16, 224, 224]
            B, T, C, H, W = vae_frames.shape

            # Compute frozen features under torch.no_grad()
            with torch.no_grad():
                # 1. Encode all frames through VAE
                # Reshape to [B*T, 3, 256, 256]
                flat_frames = vae_frames.view(B * T, C, H, W)
                latents = vae.encode(flat_frames).latent_dist.sample() * vae_scaling_factor
                latents = latents.view(B, T, 4, 32, 32)  # [B, T, 4, 32, 32]

                # 2. Extract LeV-JEPA features for the full sequence
                jepa_out = jepa_model(pixel_values=jepa_frames)
                hidden_states = jepa_out.last_hidden_state  # [B, 1 + 16*196, 1024]
                # Token partition: drop CLS prefix (token 0), remainder is 16 slots of 196 patch tokens
                patch_tokens = hidden_states[:, 1:, :]  # [B, 16*196, 1024]
                # Reshape to [B, T, 196, 1024]
                jepa_by_frame = patch_tokens.view(B, T, 196, jepa_config.embed_dim)

            # Randomly pick target transition step t from [2, ..., T-1]
            t = torch.randint(2, T, (1,)).item()

            z_curr = latents[:, t - 1]   # z_{t-1} [B, 4, 32, 32]
            z_target = latents[:, t]     # z_t     [B, 4, 32, 32]
            z_prev = latents[:, t - 2]   # z_{t-2} [B, 4, 32, 32]

            # Context latents: [z_{t-1}, z_{t-2}] concatenated -> [B, 8, 32, 32]
            # Add slight contractive latent perturbation to z_curr (sigma=0.02)
            noise_pert = torch.randn_like(z_curr) * 0.02
            z_ctx = torch.cat([z_curr + noise_pert, z_prev], dim=1)

            # JEPA guidance for target frame t
            jepa_target_tokens = jepa_by_frame[:, t]  # [B, 196, 1024]

            # Flow matching bridge formulation:
            # Sample continuous flow time s ~ U[0, 1]
            s = torch.rand(B, device=device)
            s_expanded = s.view(B, 1, 1, 1)

            # Linear bridge path between z_{t-1} (s=0) and z_t (s=1)
            z_s = (1.0 - s_expanded) * z_curr + s_expanded * z_target

            # Ground truth velocity along the straight path
            u_target = z_target - z_curr

            # Model predicts flow velocity v_theta
            v_pred = model(z_s, s, z_ctx, jepa_target_tokens)

            # Flow matching loss (L2 regression on velocity vector field)
            loss = F.mse_loss(v_pred, u_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Flow Loss: {avg_loss:.6f}")

        # Periodic rollout evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate_rollout(
                model=model,
                vae=vae,
                jepa_model=jepa_model,
                dataset=dataset,
                vae_scaling_factor=vae_scaling_factor,
                device=device,
                epoch=epoch,
                out_dir=args.out_dir,
            )


@torch.no_grad()
def evaluate_rollout(model, vae, jepa_model, dataset, vae_scaling_factor, device, epoch, out_dir):
    """Performs an autoregressive rollout of an entire 16-frame sequence and compares it to GT."""
    model.eval()
    test_item = dataset[0]  # fixed test sequence

    vae_frames = test_item["vae_frames"].unsqueeze(0).to(device)   # [1, 16, 3, 256, 256]
    jepa_frames = test_item["jepa_frames"].unsqueeze(0).to(device) # [1, 3, 16, 224, 224]
    T = vae_frames.shape[1]

    # 1. Encode ground truth sequence
    latents_gt = vae.encode(vae_frames.squeeze(0)).latent_dist.sample() * vae_scaling_factor
    latents_gt = latents_gt.unsqueeze(0)  # [1, 16, 4, 32, 32]

    # 2. Extract LeV-JEPA features for all steps
    jepa_out = jepa_model(pixel_values=jepa_frames)
    patch_tokens = jepa_out.last_hidden_state[:, 1:, :]  # [1, 16*196, 1024]
    jepa_by_frame = patch_tokens.view(1, T, 196, -1)

    # Autoregressive generation buffer starting with ground truth initial condition for t=0, 1
    rolled_latents = [latents_gt[:, 0], latents_gt[:, 1]]

    for t in range(2, T):
        z_curr = rolled_latents[-1]
        z_prev = rolled_latents[-2]
        z_ctx = torch.cat([z_curr, z_prev], dim=1)
        jepa_target = jepa_by_frame[:, t]

        # Solve ODE bridge in 2 Heun steps
        z_next = solve_flow_bridge(
            model=model,
            z_init=z_curr,
            z_ctx=z_ctx,
            jepa_tokens=jepa_target,
            steps=2,
            method="heun",
        )
        rolled_latents.append(z_next)

    rolled_latents_tensor = torch.stack(rolled_latents, dim=1)  # [1, 16, 4, 32, 32]

    # Decode latents back to pixels
    gen_frames = vae.decode(rolled_latents_tensor.squeeze(0) / vae_scaling_factor).sample  # [16, 3, 256, 256]
    gt_frames = vae_frames.squeeze(0)  # [16, 3, 256, 256]

    # Combine side-by-side: Left = Ground Truth, Right = Flow Evolution Generated
    comparison_frames = []
    for t in range(T):
        combined = torch.cat([gt_frames[t], gen_frames[t]], dim=2)  # [3, 256, 512]
        comparison_frames.append(combined)
    comparison_tensor = torch.stack(comparison_frames, dim=0)

    gif_path = os.path.join(out_dir, f"rollout_epoch_{epoch:02d}.gif")
    save_video_gif(comparison_tensor, gif_path, fps=6)
    print(f"Saved evaluation rollout comparison GIF to: {gif_path}")


if __name__ == "__main__":
    main()
