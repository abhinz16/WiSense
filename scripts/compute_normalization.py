"""Compute per-channel normalization statistics using training data only."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.settings import load_settings


def main() -> None:
    """Calculate and save training-only global channel means and standard deviations."""

    settings = load_settings()
    split_dir = settings.path("paths", "split_dir")
    dataset_dir = settings.path("paths", "dataset_dir")
    amplitude_dir = dataset_dir / settings.get("dataset", "amplitude_subdir")
    output_path = settings.path("paths", "normalization_file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(split_dir / "train.csv", dtype={"sample_id": str})
    num_channels = settings.getint("dataset", "num_channels")
    epsilon = settings.getfloat("preprocessing", "standard_deviation_epsilon")

    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sum_squared = np.zeros(num_channels, dtype=np.float64)
    channel_count = np.zeros(num_channels, dtype=np.int64)
    nonfinite_total = 0

    for index, sample_id in enumerate(train["sample_id"], start=1):
        amplitude = np.asarray(
            np.load(amplitude_dir / f"{sample_id}.npy", mmap_mode="r"), dtype=np.float64
        ).reshape(-1, num_channels)
        finite = np.isfinite(amplitude)
        nonfinite_total += int(amplitude.size - finite.sum())
        safe = np.where(finite, amplitude, 0.0)
        channel_sum += safe.sum(axis=0)
        channel_sum_squared += np.square(safe).sum(axis=0)
        channel_count += finite.sum(axis=0)

        if index % 500 == 0 or index == len(train):
            print(f"Processed {index}/{len(train)} training samples")

    if np.any(channel_count == 0):
        raise RuntimeError("At least one CSI channel has no finite training observations.")

    mean = channel_sum / channel_count
    variance = np.maximum(channel_sum_squared / channel_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    constant_channels = std < epsilon
    std[constant_channels] = 1.0

    np.savez(
        output_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=channel_count,
        constant_channel_mask=constant_channels,
        nonfinite_total=np.array(nonfinite_total, dtype=np.int64),
    )
    print("Normalization statistics written to", output_path)


if __name__ == "__main__":
    main()
