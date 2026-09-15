"""Training utilities for the temporal-pyramid sensing model."""

from __future__ import annotations

import os
import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.ml.metrics import evaluate_probabilities


IGNORE_INDEX = -100


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure Python, NumPy, PyTorch, and CUDA reproducibility settings."""

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python inside a DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a reproducibly seeded DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_multitask_targets(
    flat_targets: torch.Tensor,
    num_users: int,
    num_activities: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flattened one-hot targets into presence and activity targets."""

    matrix = flat_targets.reshape(-1, num_users, num_activities)
    presence = (matrix.sum(dim=2) > 0).float()
    activity = torch.argmax(matrix, dim=2).clone()
    activity[presence == 0] = IGNORE_INDEX
    return presence, activity


def calculate_presence_pos_weights(
    y_path,
    num_users: int,
    num_activities: int,
) -> torch.Tensor:
    """Calculate per-user BCE positive weights using training targets only."""

    targets = np.asarray(np.load(y_path, mmap_mode="r"))
    matrix = targets.reshape(-1, num_users, num_activities)
    presence = (matrix.sum(axis=2) > 0).astype(np.float32)
    positives = presence.sum(axis=0)
    negatives = presence.shape[0] - positives
    return torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    presence_criterion: nn.Module,
    activity_criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    num_users: int,
    num_activities: int,
    presence_weight: float,
    activity_weight: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    """Train for one epoch and return sample-weighted loss averages."""

    model.train()
    totals = {"loss": 0.0, "presence_loss": 0.0, "activity_loss": 0.0}
    sample_count = 0

    for inputs, flat_targets, lengths in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        flat_targets = flat_targets.to(device=device, dtype=torch.float32, non_blocking=True)
        lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
        presence_targets, activity_targets = build_multitask_targets(
            flat_targets, num_users, num_activities
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            presence_logits, activity_logits = model(inputs, lengths)
            presence_loss = presence_criterion(presence_logits, presence_targets)
            activity_loss = activity_criterion(
                activity_logits.reshape(-1, num_activities), activity_targets.reshape(-1)
            )
            loss = presence_weight * presence_loss + activity_weight * activity_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(inputs.shape[0])
        sample_count += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["presence_loss"] += float(presence_loss.item()) * batch_size
        totals["activity_loss"] += float(activity_loss.item()) * batch_size

    return {name: value / sample_count for name, value in totals.items()}


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect targets, presence probabilities, and activity probabilities."""

    model.eval()
    targets: list[np.ndarray] = []
    presence: list[np.ndarray] = []
    activity: list[np.ndarray] = []

    for inputs, flat_targets, lengths in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            presence_logits, activity_logits = model(inputs, lengths)

        targets.append(flat_targets.float().cpu().numpy())
        presence.append(torch.sigmoid(presence_logits).float().cpu().numpy())
        activity.append(torch.softmax(activity_logits, dim=2).float().cpu().numpy())

    return (
        np.concatenate(targets, axis=0),
        np.concatenate(presence, axis=0),
        np.concatenate(activity, axis=0),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_users: int,
    num_activities: int,
    presence_threshold: float,
) -> dict[str, float]:
    """Run inference over a loader and calculate the project metrics."""

    targets, presence, activity = predict_loader(model, loader, device, use_amp)
    return evaluate_probabilities(
        targets,
        presence,
        activity,
        num_users,
        num_activities,
        presence_threshold,
    )
