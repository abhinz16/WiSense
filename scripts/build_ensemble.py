"""Build the equal-weight validation ensemble from the trained seed checkpoints."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import json

import numpy as np
import pandas as pd
import torch

from src.ml.dataset import CachedCSIDataset
from src.ml.factory import load_checkpoint
from src.ml.metrics import evaluate_probabilities
from src.ml.training import make_loader, predict_loader
from src.settings import load_settings


def main() -> None:
    """Evaluate each checkpoint on validation data and save the soft ensemble."""

    settings = load_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = settings.getboolean("training", "use_amp") and device.type == "cuda"
    seeds = settings.get_int_list("training", "seeds")
    cache_dir = settings.path("paths", "cache_dir")
    model_dir = settings.path("paths", "model_dir")
    output_dir = settings.path("paths", "ensemble_results_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CachedCSIDataset(
        cache_dir / "validation_x.npy",
        cache_dir / "validation_y.npy",
        cache_dir / "validation_lengths.npy",
    )
    loader = make_loader(
        dataset,
        settings.getint("evaluation", "batch_size"),
        False,
        settings.getint("split", "random_seed"),
        settings.getint("training", "num_workers"),
        use_amp,
    )

    target_reference: np.ndarray | None = None
    presence_runs: list[np.ndarray] = []
    activity_runs: list[np.ndarray] = []
    rows: list[dict] = []

    for seed in seeds:
        model, checkpoint = load_checkpoint(model_dir / f"model_seed_{seed}.pt", device)
        targets, presence, activity = predict_loader(model, loader, device, use_amp)
        if target_reference is None:
            target_reference = targets
        elif not np.array_equal(target_reference, targets):
            raise RuntimeError("Validation targets changed between checkpoint evaluations.")

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

    assert target_reference is not None
    presence_stack = np.stack(presence_runs, axis=0)
    activity_stack = np.stack(activity_runs, axis=0)
    ensemble_presence = presence_stack.mean(axis=0)
    ensemble_activity = activity_stack.mean(axis=0)
    ensemble_metrics = evaluate_probabilities(
        target_reference,
        ensemble_presence,
        ensemble_activity,
        settings.num_users,
        settings.num_activities,
        settings.getfloat("training", "presence_threshold"),
    )

    pd.DataFrame(rows).to_csv(output_dir / "individual_metrics.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(ensemble_metrics, handle, indent=4)

    save_payload = {
        "flat_targets": target_reference,
        "ensemble_presence_probabilities": ensemble_presence,
        "ensemble_activity_probabilities": ensemble_activity,
    }
    for index, seed in enumerate(seeds):
        save_payload[f"seed_{seed}_presence_probabilities"] = presence_stack[index]
        save_payload[f"seed_{seed}_activity_probabilities"] = activity_stack[index]
    np.savez_compressed(output_dir / "validation_predictions.npz", **save_payload)

    summary = {
        "seeds": seeds,
        "ensemble_rule": "equal_weight_probability_average",
        "validation_metrics": ensemble_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print("Validation ensemble metrics")
    for name, value in ensemble_metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
