"""Build memory-mapped model inputs from WiMANS amplitude files."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd

from src.ml.labels import target_columns
from src.settings import load_settings


def numpy_dtype(name: str):
    """Translate a configured NumPy dtype name into a dtype object."""

    try:
        return np.dtype(name)
    except TypeError as error:
        raise ValueError(f"Unsupported cache dtype: {name}") from error


def preprocess_sample(
    path,
    mean: np.ndarray,
    std: np.ndarray,
    num_channels: int,
    target_length: int,
    dtype,
) -> tuple[np.ndarray, int]:
    """Normalize one amplitude sequence and pad or truncate it to the cache length."""

    amplitude = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32).reshape(-1, num_channels)
    finite = np.isfinite(amplitude)
    if not np.all(finite):
        amplitude = np.where(finite, amplitude, mean[None, :])

    normalized = (amplitude - mean[None, :]) / std[None, :]
    output = np.zeros((num_channels, target_length), dtype=dtype)
    valid_length = min(normalized.shape[0], target_length)
    output[:, :valid_length] = normalized[:valid_length].T.astype(dtype)
    return output, min(int(amplitude.shape[0]), target_length)


def build_split(name: str, settings, mean: np.ndarray, std: np.ndarray) -> None:
    """Build cached X, y, and true-length arrays for one data split."""

    split_dir = settings.path("paths", "split_dir")
    cache_dir = settings.path("paths", "cache_dir")
    dataset_dir = settings.path("paths", "dataset_dir")
    amplitude_dir = dataset_dir / settings.get("dataset", "amplitude_subdir")
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(split_dir / f"{name}.csv", dtype={"sample_id": str})
    num_channels = settings.getint("dataset", "num_channels")
    target_length = settings.getint("preprocessing", "target_length")
    dtype = numpy_dtype(settings.get("preprocessing", "cache_dtype"))
    columns = target_columns(settings.num_users, settings.activities)

    x = np.lib.format.open_memmap(
        cache_dir / f"{name}_x.npy",
        mode="w+",
        dtype=dtype,
        shape=(len(frame), num_channels, target_length),
    )
    y = np.lib.format.open_memmap(
        cache_dir / f"{name}_y.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(frame), len(columns)),
    )
    lengths = np.lib.format.open_memmap(
        cache_dir / f"{name}_lengths.npy",
        mode="w+",
        dtype=np.int32,
        shape=(len(frame),),
    )

    for index, row in frame.iterrows():
        sample, length = preprocess_sample(
            amplitude_dir / f"{row['sample_id']}.npy",
            mean,
            std,
            num_channels,
            target_length,
            dtype,
        )
        x[index] = sample
        y[index] = row[columns].to_numpy(dtype=np.uint8)
        lengths[index] = length
        if (index + 1) % 500 == 0 or (index + 1) == len(frame):
            print(f"{name}: {index + 1}/{len(frame)}")

    x.flush()
    y.flush()
    lengths.flush()


def main() -> None:
    """Build caches for train, validation, and test splits."""

    settings = load_settings()
    stats = np.load(settings.path("paths", "normalization_file"))
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

    for split in ("train", "validation", "test"):
        build_split(split, settings, mean, std)

    cache_dir = settings.path("paths", "cache_dir")
    summary = {
        "normalization": "training_global_per_channel_z_score",
        "target_length": settings.getint("preprocessing", "target_length"),
        "num_channels": settings.getint("dataset", "num_channels"),
        "cache_dtype": settings.get("preprocessing", "cache_dtype"),
    }
    with open(cache_dir / "cache_config.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
    print("Tensor cache written to", cache_dir)


if __name__ == "__main__":
    main()
