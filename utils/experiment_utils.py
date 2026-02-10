import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.data_utils import ECGDenoisingDataset, get_optimized_dataset


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42


@dataclass
class DataConfig:
    data_path: str = "./data/"
    sampling_rate: int = 100
    noise_all: bool = True
    fs: int = 100
    noise_snr_range: Tuple[int, int] = (5, 25)
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_dataset(data_cfg: DataConfig) -> ECGDenoisingDataset:
    x_all = get_optimized_dataset(
        data_cfg.data_path,
        sampling_rate=data_cfg.sampling_rate,
        preprocess=False,
        n_jobs=-1,
    )
    return ECGDenoisingDataset(
        data=x_all,
        noise_all=data_cfg.noise_all,
        fs=data_cfg.fs,
        noise_snr_range=data_cfg.noise_snr_range,
    )


def _get_split_file(data_path: str, split_cfg: SplitConfig) -> str:
    fname = f"splits_seed{split_cfg.seed}_v1.npz"
    return os.path.join(data_path, fname)


def get_or_create_split_indices(
    data_len: int,
    data_path: str,
    split_cfg: SplitConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_file = _get_split_file(data_path, split_cfg)
    if os.path.exists(split_file):
        splits = np.load(split_file)
        return splits["train_idx"], splits["val_idx"], splits["test_idx"]

    indices = np.arange(data_len)
    rng = np.random.RandomState(split_cfg.seed)
    rng.shuffle(indices)

    train_end = int(split_cfg.train_ratio * data_len)
    val_end = train_end + int(split_cfg.val_ratio * data_len)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    os.makedirs(data_path, exist_ok=True)
    np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return train_idx, val_idx, test_idx


def build_dataloaders(
    data_cfg: DataConfig,
    split_cfg: SplitConfig,
):
    full_dataset = load_full_dataset(data_cfg)
    train_idx, val_idx, test_idx = get_or_create_split_indices(
        len(full_dataset),
        data_cfg.data_path,
        split_cfg,
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
