"""Train the final temporal-pyramid model for each configured random seed."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.ml.dataset import CachedCSIDataset
from src.ml.factory import build_model, load_best_values, model_config_from_values
from src.ml.model import count_trainable_parameters
from src.ml.training import calculate_presence_pos_weights, configure_reproducibility, evaluate_model, make_loader, train_epoch
from src.settings import load_settings


def checkpoint_path(model_dir: Path, seed: int) -> Path:
    """Return the checkpoint filename for one training seed."""

    return model_dir / f"model_seed_{seed}.pt"


def train_seed(seed: int, settings, values: dict) -> dict[str, float | int]:
    """Train one seeded model and save the best validation checkpoint."""

    configure_reproducibility(seed, settings.getboolean("training", "deterministic"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = settings.getboolean("training", "use_amp") and device.type == "cuda"
    cache_dir = settings.path("paths", "cache_dir")
    model_dir = settings.path("paths", "model_dir")
    history_dir = settings.path("paths", "training_results_dir") / "histories"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CachedCSIDataset(
        cache_dir / "train_x.npy",
        cache_dir / "train_y.npy",
        cache_dir / "train_lengths.npy",
    )
    validation_dataset = CachedCSIDataset(
        cache_dir / "validation_x.npy",
        cache_dir / "validation_y.npy",
        cache_dir / "validation_lengths.npy",
    )

    batch_size = int(values["batch_size"])
    workers = settings.getint("training", "num_workers")
    train_loader = make_loader(train_dataset, batch_size, True, seed, workers, use_amp)
    validation_loader = make_loader(validation_dataset, batch_size, False, seed, workers, use_amp)

    model_config = model_config_from_values(settings, values)
    model = build_model(model_config).to(device)
    pos_weight = calculate_presence_pos_weights(
        cache_dir / "train_y.npy", settings.num_users, settings.num_activities
    ).to(device)
    presence_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    activity_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=settings.getfloat("training", "scheduler_factor"),
        patience=settings.getint("training", "scheduler_patience"),
        min_lr=settings.getfloat("training", "minimum_learning_rate"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_epochs = settings.getint("training", "max_epochs")
    patience = settings.getint("training", "early_stopping_patience")
    min_delta = settings.getfloat("training", "minimum_improvement")
    threshold = settings.getfloat("training", "presence_threshold")
    presence_weight = settings.getfloat("training", "presence_loss_weight")
    activity_weight = float(values["activity_loss_weight"])
    gradient_clip = settings.getfloat("training", "gradient_clip_norm")

    best_score = -np.inf
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    no_improvement = 0
    history: list[dict] = []
    output_path = checkpoint_path(model_dir, seed)
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            presence_criterion,
            activity_criterion,
            scaler,
            device,
            use_amp,
            settings.num_users,
            settings.num_activities,
            presence_weight,
            activity_weight,
            gradient_clip,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            use_amp,
            settings.num_users,
            settings.num_activities,
            threshold,
        )
        score = float(validation_metrics["activity_macro_f1"])
        scheduler.step(score)
        improved = score > best_score + min_delta

        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            no_improvement = 0
            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model_config": model_config,
                    "selected_hyperparameters": values,
                    "training_config": {
                        "presence_threshold": threshold,
                        "presence_loss_weight": presence_weight,
                        "activity_loss_weight": activity_weight,
                        "gradient_clip_norm": gradient_clip,
                    },
                    "validation_metrics": best_metrics,
                    "model_state_dict": model.state_dict(),
                },
                output_path,
            )
        else:
            no_improvement += 1

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_dir / f"seed_{seed}.csv", index=False)

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"seed {seed} | epoch {epoch:03d} | "
                f"activity F1 {score:.4f} | best {best_score:.4f} | "
                f"presence F1 {validation_metrics['presence_macro_f1']:.4f}"
            )

        if no_improvement >= patience:
            print(f"seed {seed}: early stopping at epoch {epoch}")
            break

    if best_metrics is None:
        raise RuntimeError(f"Training seed {seed} did not produce a valid checkpoint.")

    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        **best_metrics,
        "trainable_parameters": count_trainable_parameters(model),
        "training_minutes": float((time.perf_counter() - start) / 60.0),
    }

    del model, optimizer, scaler, train_loader, validation_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    """Train every configured seed and save a compact training summary."""

    settings = load_settings()
    values = load_best_values(settings)
    result_dir = settings.path("paths", "training_results_dir")
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = [train_seed(seed, settings, values) for seed in settings.get_int_list("training", "seeds")]
    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "seed_metrics.csv", index=False)

    metrics = [
        "presence_macro_f1",
        "activity_top1_accuracy",
        "activity_macro_f1",
        "structured_macro_f1",
        "structured_accuracy",
        "absent_user_fpr",
    ]
    summary = {
        "seeds": settings.get_int_list("training", "seeds"),
        "selected_hyperparameters_file": str(settings.path("paths", "best_values_file")),
        "metrics": {
            metric: {
                "mean": float(frame[metric].mean()),
                "sample_std": float(frame[metric].std(ddof=1)),
            }
            for metric in metrics
        },
    }
    with open(result_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print("\nTraining complete")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
