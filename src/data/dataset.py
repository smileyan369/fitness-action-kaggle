"""Window datasets for pose-sequence classification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.build import make_windows


class WindowDataset(Dataset):
    """Sliding windows over one or more processed sequences.

    manifest columns: video_id, label, seq_path, split
    Each seq .npz contains `features` (T,71).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_root: str | Path,
        class_to_idx: dict[str, int],
        seq_len: int = 48,
        stride: int = 8,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.class_to_idx = class_to_idx
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None
        self.augment = augment
        self.return_meta = False
        self.items: list[tuple[np.ndarray, int, str, int]] = []  # window, label, video_id, start

        for row in manifest.itertuples(index=False):
            path = self.data_root / row.seq_path
            arr = np.load(path)
            seq = arr["features"].astype(np.float32)
            windows, starts = make_windows(seq, seq_len=seq_len, stride=stride)
            label = class_to_idx[row.label]
            for w, s in zip(windows, starts):
                self.items.append((w, label, str(row.video_id), int(s)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        window, label, video_id, start = self.items[idx]
        x = window.astype(np.float32, copy=True)
        if self.augment:
            x = augment_window(x)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        x = torch.from_numpy(x)
        y = torch.tensor(label, dtype=torch.long)
        if self.return_meta:
            return x, y, video_id, start
        return x, y


def compute_scaler(dataset: WindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from all windows in a dataset."""
    n = 0
    s = np.zeros(71, dtype=np.float64)
    ss = np.zeros(71, dtype=np.float64)
    for window, _, _, _ in dataset.items:
        w = window.astype(np.float64)
        n += w.shape[0]
        s += w.sum(axis=0)
        ss += (w * w).sum(axis=0)
    mean = s / max(n, 1)
    var = np.maximum(ss / max(n, 1) - mean * mean, 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def augment_window(x: np.ndarray) -> np.ndarray:
    """Light augmentation for 71-D features.

    - random temporal reverse (actions are mostly symmetric in time for
      classification; use only sometimes)
    - coordinate noise
    - horizontal mirror (swap left/right features)
    """
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        x = mirror_features(x)
    if np.random.rand() < 0.2:
        x = x[::-1].copy()
    return x.astype(np.float32)


def mirror_features(x: np.ndarray) -> np.ndarray:
    """Mirror 71-D features horizontally (swap COCO left/right)."""
    y = x.copy()
    # (T,17,3): negate x, swap left/right keypoints
    k = y[:, :51].reshape(-1, 17, 3)
    k[:, :, 0] *= -1.0
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    for a, b in pairs:
        k[:, [a, b]] = k[:, [b, a]]
    y[:, :51] = k.reshape(-1, 51)
    # angles: 51..60 are 5 left/right pairs; 61 torso, 62 shoulder tilt unchanged
    for a, b in [(51, 52), (53, 54), (55, 56), (57, 58), (59, 60)]:
        y[:, [a, b]] = y[:, [b, a]]
    # distances: 63 shoulder width, 64 hip width, 65 torso,
    # 66/67 hip-knee, 68/69 knee-ankle, 70 wrist-shoulder mean
    for a, b in [(66, 67), (68, 69)]:
        y[:, [a, b]] = y[:, [b, a]]
    return y
