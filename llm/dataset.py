import os
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MemmapTextDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenized binary text data (int32 or uint16).
    Reads sequences directly from disk with O(1) RAM footprint.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int = 256,
        dtype: np.dtype = np.int32,
    ):
        super().__init__()
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at: {bin_path}")

        self.bin_path = bin_path
        self.seq_len = seq_len
        self.dtype = dtype

        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        # Each sample returns seq_len + 1 tokens so x = tokens[:-1] and y = tokens[1:]
        self.num_samples = (len(self.data) - 1) // seq_len

        if self.num_samples == 0:
            raise ValueError(
                f"File {bin_path} contains {len(self.data)} tokens, which is smaller than seq_len={seq_len}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end].astype(np.int64)
        return torch.from_numpy(chunk)


def get_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    seq_len: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates high-throughput DataLoaders for training and validation datasets.
    """
    train_dataset = MemmapTextDataset(train_path, seq_len=seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_path and os.path.exists(val_path):
        val_dataset = MemmapTextDataset(val_path, seq_len=seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader
