"""Model construction and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.ml.model import TemporalPyramidModel
from src.settings import ProjectSettings


def load_best_values(settings: ProjectSettings) -> dict[str, Any]:
    """Read the selected hyperparameters from the configured JSON artifact."""

    path = settings.path("paths", "best_values_file")
    if not path.exists():
        raise FileNotFoundError(
            f"Selected hyperparameter file not found: {path}. "
            "The repository includes this artifact so the published model can be reproduced."
        )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} does not contain a 'parameters' object.")
    return parameters


def model_config_from_values(settings: ProjectSettings, values: dict[str, Any]) -> dict[str, Any]:
    """Combine dataset/model settings with the selected hyperparameters."""

    required = (
        "base_channels",
        "num_blocks",
        "kernel_size",
        "cnn_dropout",
        "lstm_hidden_size",
        "lstm_layers",
        "lstm_dropout",
        "head_dropout",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise KeyError(f"Selected hyperparameters are missing: {', '.join(missing)}")

    return {
        "input_channels": settings.getint("dataset", "num_channels"),
        "base_channels": int(values["base_channels"]),
        "num_blocks": int(values["num_blocks"]),
        "kernel_size": int(values["kernel_size"]),
        "cnn_dropout": float(values["cnn_dropout"]),
        "lstm_hidden_size": int(values["lstm_hidden_size"]),
        "lstm_layers": int(values["lstm_layers"]),
        "lstm_dropout": float(values["lstm_dropout"]),
        "head_dropout": float(values["head_dropout"]),
        "num_users": settings.num_users,
        "num_activities": settings.num_activities,
        "temporal_segments": settings.getint("model", "temporal_segments"),
        "max_cnn_channels": settings.getint("model", "max_cnn_channels"),
        "attention_hidden_min": settings.getint("model", "attention_hidden_min"),
    }


def build_model(model_config: dict[str, Any]) -> TemporalPyramidModel:
    """Instantiate the production architecture from a saved model configuration."""

    return TemporalPyramidModel(**model_config)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[TemporalPyramidModel, dict[str, Any]]:
    """Load a trusted local checkpoint and return an evaluation-ready model."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint
