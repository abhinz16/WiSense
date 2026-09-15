"""Run the single locked final evaluation on the test split."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from src.ml.dataset import CachedCSIDataset
from src.ml.factory import load_checkpoint
from src.ml.metrics import activity_confusion_matrix, evaluate_probabilities, per_class_activity_metrics
from src.ml.training import make_loader, predict_loader
from src.settings import load_settings


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it fully into memory."""

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            hasher.update(block)
    return hasher.hexdigest()


def write_lock(settings, checkpoint_paths: list[Path], output_dir: Path) -> dict:
    """Record the exact checkpoint files and decision rules before reading test data."""

    lock = {
        "status": "frozen_before_test_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "presence_threshold": settings.getfloat("training", "presence_threshold"),
        "ensemble_rule": "equal_weight_probability_average",
        "ensemble_weights": [1.0 / len(checkpoint_paths)] * len(checkpoint_paths),
        "seeds": settings.get_int_list("training", "seeds"),
        "checkpoints": [
            {"path": str(path.relative_to(settings.project_root)), "sha256": sha256_file(path)}
            for path in checkpoint_paths
        ],
        "post_test_rule": "Do not tune preprocessing, model settings, thresholds, ensemble weights, or checkpoints using test results.",
    }
    with open(output_dir / "model_lock.json", "w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=4)
    return lock


def main() -> None:
    """Freeze the selected ensemble, evaluate it once, and save reporting artifacts."""

    settings = load_settings()
    output_dir = settings.path("paths", "final_results_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    if metrics_path.exists() and settings.getboolean("evaluation", "prevent_test_overwrite"):
        raise RuntimeError(
            f"Final test metrics already exist at {metrics_path}. "
            "Delete the final-results directory deliberately if a full clean rerun is intended."
        )

    seeds = settings.get_int_list("training", "seeds")
    model_dir = settings.path("paths", "model_dir")
    checkpoint_paths = [model_dir / f"model_seed_{seed}.pt" for seed in seeds]
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    lock = write_lock(settings, checkpoint_paths, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = settings.getboolean("training", "use_amp") and device.type == "cuda"
    cache_dir = settings.path("paths", "cache_dir")
    dataset = CachedCSIDataset(
        cache_dir / "test_x.npy",
        cache_dir / "test_y.npy",
        cache_dir / "test_lengths.npy",
    )
    loader = make_loader(
        dataset,
        settings.getint("evaluation", "batch_size"),
        False,
        settings.getint("split", "random_seed"),
        settings.getint("training", "num_workers"),
        use_amp,
    )

    targets_reference: np.ndarray | None = None
    presence_runs: list[np.ndarray] = []
    activity_runs: list[np.ndarray] = []
    rows: list[dict] = []

    for seed, path in zip(seeds, checkpoint_paths, strict=True):
        model, checkpoint = load_checkpoint(path, device)
        targets, presence, activity = predict_loader(model, loader, device, use_amp)
        if targets_reference is None:
            targets_reference = targets
        elif not np.array_equal(targets_reference, targets):
            raise RuntimeError("Test targets changed between checkpoint evaluations.")

        metrics = evaluate_probabilities(
            targets,
            presence,
            activity,
            settings.num_users,
            settings.num_activities,
            settings.getfloat("training", "presence_threshold"),
        )
        rows.append({"seed": seed, "checkpoint_epoch": int(checkpoint["epoch"]), **metrics})
        presence_runs.append(presence)
        activity_runs.append(activity)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert targets_reference is not None
    presence_stack = np.stack(presence_runs, axis=0)
    activity_stack = np.stack(activity_runs, axis=0)
    weights = np.full(len(seeds), 1.0 / len(seeds), dtype=np.float64)
    ensemble_presence = np.average(presence_stack, axis=0, weights=weights)
    ensemble_activity = np.average(activity_stack, axis=0, weights=weights)
    metrics = evaluate_probabilities(
        targets_reference,
        ensemble_presence,
        ensemble_activity,
        settings.num_users,
        settings.num_activities,
        settings.getfloat("training", "presence_threshold"),
    )

    pd.DataFrame(rows).to_csv(output_dir / "individual_metrics.csv", index=False)
    per_class_activity_metrics(
        targets_reference, ensemble_activity, settings.num_users, settings.activities
    ).to_csv(output_dir / "per_class_metrics.csv", index=False)
    activity_confusion_matrix(
        targets_reference, ensemble_activity, settings.num_users, settings.activities
    ).to_csv(output_dir / "confusion_matrix.csv")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4)

    save_payload = {
        "flat_targets": targets_reference,
        "ensemble_presence_probabilities": ensemble_presence,
        "ensemble_activity_probabilities": ensemble_activity,
    }
    for index, seed in enumerate(seeds):
        save_payload[f"seed_{seed}_presence_probabilities"] = presence_stack[index]
        save_payload[f"seed_{seed}_activity_probabilities"] = activity_stack[index]
    np.savez_compressed(output_dir / "predictions.npz", **save_payload)

    validation_metrics = None
    validation_path = settings.path("paths", "ensemble_results_dir") / "metrics.json"
    if validation_path.exists():
        with open(validation_path, "r", encoding="utf-8") as handle:
            validation_metrics = json.load(handle)

    report = {
        "lock": lock,
        "test_samples": int(len(dataset)),
        "ensemble_metrics": metrics,
        "validation_reference": validation_metrics,
        "test_minus_validation": {
            key: float(metrics[key] - validation_metrics[key])
            for key in metrics
            if validation_metrics is not None and key in validation_metrics
        },
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    print("Final test metrics")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
