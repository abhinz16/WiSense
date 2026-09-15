"""Memory-mapped dataset for cached CSI tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedCSIDataset(Dataset):
    """Expose cached CSI, targets, and true packet lengths to PyTorch."""

    def __init__(self, x_path: str | Path, y_path: str | Path, lengths_path: str | Path):
        """Validate cache files without loading the full arrays into memory."""

        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.lengths_path = Path(lengths_path)

        for path in (self.x_path, self.y_path, self.lengths_path):
            if not path.exists():
                raise FileNotFoundError(path)

        x = np.load(self.x_path, mmap_mode="r")
        y = np.load(self.y_path, mmap_mode="r")
        lengths = np.load(self.lengths_path, mmap_mode="r")

        if x.ndim != 3:
            raise ValueError(f"Expected X with shape [N, C, T], got {x.shape}")
        if x.shape[0] != y.shape[0] or x.shape[0] != lengths.shape[0]:
            raise ValueError("Cached X, y, and length arrays do not contain the same number of samples.")

        self.num_samples = int(x.shape[0])
        self.input_shape = tuple(x.shape[1:])
        self.max_length = int(x.shape[2])
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._lengths: np.ndarray | None = None

    def _ensure_open(self) -> None:
        """Open memory maps lazily so DataLoader workers create their own handles."""

        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="c")
        if self._y is None:
            self._y = np.load(self.y_path, mmap_mode="c")
        if self._lengths is None:
            self._lengths = np.load(self.lengths_path, mmap_mode="r")

    def __len__(self) -> int:
        """Return the number of cached samples."""

        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one cached CSI tensor, flattened target, and true packet length."""

        self._ensure_open()
        assert self._x is not None and self._y is not None and self._lengths is not None

        x = torch.from_numpy(self._x[index])
        y = torch.from_numpy(self._y[index].astype(np.float32, copy=True))
        length = max(1, min(int(self._lengths[index]), self.max_length))
        return x, y, torch.tensor(length, dtype=torch.long)
