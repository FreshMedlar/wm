"""Toy Video Dataset: Moving Objects / Bouncing Geometry.

Generates synthetic video sequences with geometric shapes (circles, squares,
rectangles) bouncing, translating, occluding, and disoccluding with randomized
colors, textures, and velocities. Provides both on-the-fly rendering and tensor
formatting for LeV-JEPA (224x224, ImageNet norm) and VAE (256x256, [-1, 1] range).
"""

import math
import random
import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticBouncingShapesDataset(Dataset):
    """Generates continuous motion video sequences with geometric shapes bouncing inside a canvas.

    Each item yields:
      - vae_frames: [T, 3, 256, 256] in range [-1.0, 1.0] for VAE encoding/decoding.
      - jepa_frames: [3, T, 224, 224] normalized with ImageNet stats for LeV-JEPA.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 16,
        num_objects: int = 3,
        vae_size: int = 256,
        jepa_size: int = 224,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_objects = num_objects
        self.vae_size = vae_size
        self.jepa_size = jepa_size

        # ImageNet normalization stats
        self.register_norm = {
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1),
        }

    def __len__(self) -> int:
        return self.num_samples

    def _render_sequence(self, seed: int):
        rng = np.random.RandomState(seed)

        # Background color: subtle dark gradient / shade
        bg_val = rng.uniform(0.05, 0.25)
        frames = np.full(
            (self.seq_length, self.vae_size, self.vae_size, 3),
            bg_val,
            dtype=np.float32,
        )

        # Initialize object states: pos [x, y], vel [vx, vy], radius, color, shape_type
        objects = []
        for _ in range(self.num_objects):
            radius = rng.uniform(18.0, 36.0)
            x = rng.uniform(radius, self.vae_size - radius)
            y = rng.uniform(radius, self.vae_size - radius)
            speed = rng.uniform(3.0, 8.0)
            angle = rng.uniform(0, 2 * math.pi)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            color = rng.uniform(0.4, 0.95, size=3).astype(np.float32)
            shape = rng.choice(["circle", "square"])
            objects.append(
                {
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "r": radius,
                    "color": color,
                    "shape": shape,
                }
            )

        yy, xx = np.mgrid[0 : self.vae_size, 0 : self.vae_size]

        for t in range(self.seq_length):
            frame = frames[t]
            for obj in objects:
                # Update position and bounce off canvas boundaries
                obj["x"] += obj["vx"]
                obj["y"] += obj["vy"]

                if obj["x"] - obj["r"] < 0:
                    obj["x"] = obj["r"]
                    obj["vx"] *= -1
                elif obj["x"] + obj["r"] >= self.vae_size:
                    obj["x"] = self.vae_size - 1 - obj["r"]
                    obj["vx"] *= -1

                if obj["y"] - obj["r"] < 0:
                    obj["y"] = obj["r"]
                    obj["vy"] *= -1
                elif obj["y"] + obj["r"] >= self.vae_size:
                    obj["y"] = self.vae_size - 1 - obj["r"]
                    obj["vy"] *= -1

                # Render shape mask
                if obj["shape"] == "circle":
                    mask = (xx - obj["x"]) ** 2 + (yy - obj["y"]) ** 2 <= obj["r"] ** 2
                else:
                    mask = (
                        (np.abs(xx - obj["x"]) <= obj["r"])
                        & (np.abs(yy - obj["y"]) <= obj["r"])
                    )

                frame[mask] = obj["color"]

        return frames

    def __getitem__(self, idx: int):
        # Deterministic per index for reproducibility
        seed = (idx * 9973 + 12345) % (2**31 - 1)
        raw_frames = self._render_sequence(seed)  # [T, H, W, 3] in [0, 1]

        # Convert to torch tensor: [T, 3, H, W]
        frames_t = torch.from_numpy(raw_frames).permute(0, 3, 1, 2)  # [T, 3, 256, 256]

        # VAE frames: normalized to [-1.0, 1.0]
        vae_frames = frames_t * 2.0 - 1.0

        # LeV-JEPA frames: resize to 224x224, shape [3, T, 224, 224], ImageNet norm
        # PyTorch interpolate expects [N, C, H, W]
        jepa_resized = torch.nn.functional.interpolate(
            frames_t,
            size=(self.jepa_size, self.jepa_size),
            mode="bilinear",
            align_corners=False,
        )  # [T, 3, 224, 224]
        jepa_video = jepa_resized.permute(1, 0, 2, 3)  # [3, T, 224, 224]

        # ImageNet normalization
        mean = self.register_norm["mean"]
        std = self.register_norm["std"]
        jepa_video = (jepa_video - mean) / std

        return {
            "vae_frames": vae_frames,      # [T, 3, 256, 256] in [-1, 1]
            "jepa_frames": jepa_video,     # [3, T, 224, 224] ImageNet normalized
        }
