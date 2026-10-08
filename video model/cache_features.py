"""Offline Precomputation & Feature Caching for LeV-JEPA and VAE.

Extracts and caches latent representations to disk:
1. VAE latents [T, 4, 32, 32]
2. LeV-JEPA patch tokens [T, 196, 1024]
This eliminates all VRAM usage from the frozen models during DiT training
and accelerates training throughput by ~10x.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/medlar/Projects")

from diffusers import AutoencoderKL
from transformers import AutoModel, AutoConfig

from dataset import SyntheticBouncingShapesDataset


def cache_dataset(
    num_samples: int = 200,
    seq_length: int = 16,
    batch_size: int = 1,
    cache_dir: str = str(SCRIPT_DIR / "latent_cache"),
    device: str = "cuda:0",
):
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Caching {num_samples} video sequences to {cache_dir} on {device}...")

    # Load frozen VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("/home/medlar/Projects/sd-vae-ft-mse", local_files_only=True)
    vae = vae.to(device).eval()
    vae_scaling_factor = vae.config.scaling_factor or 0.18215

    # Load frozen LeV-JEPA
    print("Loading LeV-JEPA...")
    jepa_config = AutoConfig.from_pretrained("/home/medlar/Projects/LeVJEPA", trust_remote_code=True, local_files_only=True)
    jepa_model = AutoModel.from_pretrained("/home/medlar/Projects/LeVJEPA", config=jepa_config, trust_remote_code=True, local_files_only=True)
    jepa_model = jepa_model.to(device).eval()

    dataset = SyntheticBouncingShapesDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        num_objects=3,
        vae_size=256,
        jepa_size=224,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Precomputing features"):
            vae_frames = batch["vae_frames"].to(device)   # [B, T, 3, 256, 256]
            jepa_frames = batch["jepa_frames"].to(device) # [B, 3, T, 224, 224]
            B, T, C, H, W = vae_frames.shape

            # Encode VAE frame-by-frame to keep VRAM minimal (< 1 GB)
            latents_list = []
            for t in range(T):
                f_t = vae_frames[:, t]  # [B, 3, 256, 256]
                lat = vae.encode(f_t).latent_dist.sample() * vae_scaling_factor  # [B, 4, 32, 32]
                latents_list.append(lat.cpu())
            latents = torch.stack(latents_list, dim=1)  # [B, T, 4, 32, 32]

            # LeV-JEPA extraction for the clip
            jepa_out = jepa_model(pixel_values=jepa_frames)
            patch_tokens = jepa_out.last_hidden_state[:, 1:, :]  # [B, T*196, 1024]
            jepa_by_frame = patch_tokens.view(B, T, 196, jepa_config.embed_dim).cpu()

            # Save individual samples to disk
            for b in range(B):
                save_data = {
                    "latents": latents[b].clone(),            # [T, 4, 32, 32]
                    "jepa_tokens": jepa_by_frame[b].clone(),  # [T, 196, 1024]
                }
                torch.save(save_data, os.path.join(cache_dir, f"sample_{sample_idx:05d}.pt"))
                sample_idx += 1

    print(f"Successfully cached {sample_idx} sequences to: {cache_dir}")


class CachedLatentDataset(Dataset):
    """Fast dataset that loads precomputed VAE latents and LeV-JEPA tokens from disk."""

    def __init__(self, cache_dir: str = str(SCRIPT_DIR / "latent_cache")):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(list(self.cache_dir.glob("sample_*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No cached samples found in {cache_dir}. Run cache_features.py first!")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--cache-dir", type=str, default=str(SCRIPT_DIR / "latent_cache"))
    args = parser.parse_args()
    cache_dataset(num_samples=args.num_samples, seq_length=args.seq_len, cache_dir=args.cache_dir)
