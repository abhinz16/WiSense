"""Lazy production inference for the frozen temporal-pyramid ensemble."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.ml.factory import load_checkpoint
from src.settings import ProjectSettings, load_settings


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to JSON-safe values."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class WiSenseInferenceEngine:
    """Load the configured seed ensemble on demand and serve sample predictions."""

    def __init__(self, settings: ProjectSettings | None = None) -> None:
        """Prepare lazy model, split-cache, and verification state."""

        self.settings = settings or load_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.settings.getboolean("training", "use_amp") and self.device.type == "cuda"
        self.presence_threshold = self.settings.getfloat("training", "presence_threshold")
        self.seeds = self.settings.get_int_list("training", "seeds")
        self.model_dir = self.settings.path("paths", "model_dir")
        self.cache_dir = self.settings.path("paths", "cache_dir")
        self.split_dir = self.settings.path("paths", "split_dir")
        self.models: dict[int, torch.nn.Module] = {}
        self.checkpoints: dict[int, dict[str, Any]] = {}
        self.model_loaded = False
        self.model_load_seconds: float | None = None
        self.split_cache: dict[str, dict[str, Any]] = {}
        self.frozen_prediction_cache: dict[str, dict[str, Any]] = {}
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _checkpoint_path(self, seed: int) -> Path:
        """Return the configured checkpoint path for one ensemble seed."""

        return self.model_dir / f"model_seed_{seed}.pt"

    def _ensure_models_loaded(self) -> bool:
        """Load all seed checkpoints once and return whether this call did the loading."""

        if self.model_loaded:
            return False
        with self._model_lock:
            if self.model_loaded:
                return False
            start = time.perf_counter()
            for seed in self.seeds:
                model, checkpoint = load_checkpoint(self._checkpoint_path(seed), self.device)
                self.models[seed] = model
                self.checkpoints[seed] = checkpoint
            self.model_load_seconds = float(time.perf_counter() - start)
            self.model_loaded = True
            return True

    def _ensure_split_loaded(self, split: str) -> dict[str, Any]:
        """Memory-map one processed split and its manifest."""

        split = split.lower().strip()
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if split in self.split_cache:
            return self.split_cache[split]

        paths = {
            "x": self.cache_dir / f"{split}_x.npy",
            "y": self.cache_dir / f"{split}_y.npy",
            "lengths": self.cache_dir / f"{split}_lengths.npy",
            "metadata": self.split_dir / f"{split}.csv",
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

        metadata = pd.read_csv(paths["metadata"], dtype={"sample_id": str})
        payload = {
            "x": np.load(paths["x"], mmap_mode="r"),
            "y": np.load(paths["y"], mmap_mode="r"),
            "lengths": np.load(paths["lengths"], mmap_mode="r"),
            "metadata": metadata,
            "sample_to_row": {sample_id: index for index, sample_id in enumerate(metadata["sample_id"].astype(str))},
        }
        if not (len(payload["x"]) == len(payload["y"]) == len(payload["lengths"]) == len(metadata)):
            raise RuntimeError(f"Processed split '{split}' is not row-aligned.")
        self.split_cache[split] = payload
        return payload

    def _row_for_sample(self, sample_id: str, split: str) -> tuple[int, pd.Series]:
        """Return the processed row index and metadata for a raw sample ID."""

        data = self._ensure_split_loaded(split)
        key = str(sample_id)
        if key not in data["sample_to_row"]:
            raise KeyError(f"Sample '{sample_id}' was not found in split '{split}'.")
        row_index = int(data["sample_to_row"][key])
        return row_index, data["metadata"].iloc[row_index]

    @torch.inference_mode()
    def _infer_row(self, split: str, row_index: int) -> dict[str, Any]:
        """Run all seed models for one cached sample and average their probabilities."""

        loaded_now = self._ensure_models_loaded()
        data = self._ensure_split_loaded(split)
        sample = np.asarray(data["x"][row_index], dtype=np.float32)
        length = max(1, min(int(data["lengths"][row_index]), sample.shape[1]))
        x = torch.from_numpy(sample.copy()).unsqueeze(0).to(self.device, dtype=torch.float32)
        lengths = torch.tensor([length], device=self.device, dtype=torch.long)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        presence_runs: list[np.ndarray] = []
        activity_runs: list[np.ndarray] = []

        with self._inference_lock:
            for seed in self.seeds:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    presence_logits, activity_logits = self.models[seed](x, lengths)
                presence_runs.append(torch.sigmoid(presence_logits).float().cpu().numpy()[0])
                activity_runs.append(torch.softmax(activity_logits, dim=2).float().cpu().numpy()[0])

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = float((time.perf_counter() - start) * 1000.0)
        presence_stack = np.stack(presence_runs, axis=0)
        activity_stack = np.stack(activity_runs, axis=0)
        return {
            "presence_probabilities": presence_stack.mean(axis=0, dtype=np.float64).astype(np.float32),
            "activity_probabilities": activity_stack.mean(axis=0, dtype=np.float64).astype(np.float32),
            "per_model_presence_probabilities": presence_stack,
            "per_model_activity_probabilities": activity_stack,
            "true_length": length,
            "input_shape": list(x.shape),
            "model_loaded_now": loaded_now,
            "model_load_seconds": self.model_load_seconds,
            "inference_ms": inference_ms,
        }

    def predict_sample(self, sample_id: str, split: str = "test", top_k: int = 3) -> dict[str, Any]:
        """Return human-readable ensemble predictions and ground truth for one sample."""

        split = split.lower().strip()
        row_index, metadata_row = self._row_for_sample(sample_id, split)
        raw = self._infer_row(split, row_index)
        data = self._ensure_split_loaded(split)
        target = np.asarray(data["y"][row_index], dtype=np.float32).reshape(
            self.settings.num_users, self.settings.num_activities
        )
        true_presence = target.sum(axis=1) > 0
        true_activity = np.argmax(target, axis=1)
        top_k = max(1, min(int(top_k), self.settings.num_activities))

        users: list[dict[str, Any]] = []
        for user_index in range(self.settings.num_users):
            presence_probability = float(raw["presence_probabilities"][user_index])
            predicted_present = presence_probability >= self.presence_threshold
            probabilities = raw["activity_probabilities"][user_index]
            activity_index = int(np.argmax(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            actual_present = bool(true_presence[user_index])
            users.append(
                {
                    "user_index": user_index + 1,
                    "predicted_present": bool(predicted_present),
                    "presence_probability": presence_probability,
                    "activity_argmax": activity_index,
                    "activity_argmax_probability": float(probabilities[activity_index]),
                    "predicted_activity": self.settings.activities[activity_index] if predicted_present else None,
                    "top_activities": [
                        {
                            "activity_index": int(index),
                            "activity": self.settings.activities[int(index)],
                            "probability": float(probabilities[int(index)]),
                        }
                        for index in top_indices
                    ],
                    "activity_probabilities": {
                        name: float(probability)
                        for name, probability in zip(self.settings.activities, probabilities, strict=True)
                    },
                    "true_present": actual_present,
                    "true_activity": self.settings.activities[int(true_activity[user_index])] if actual_present else None,
                    "true_activity_index": int(true_activity[user_index]) if actual_present else None,
                }
            )

        metadata = {str(key): _json_safe(value) for key, value in metadata_row.items()}
        return {
            "sample_id": str(sample_id),
            "split": split,
            "row_index": row_index,
            "model": "Temporal-Pyramid Ensemble",
            "device": str(self.device),
            "presence_threshold": self.presence_threshold,
            "ensemble_size": len(self.seeds),
            "ensemble_seeds": self.seeds,
            "ensemble_rule": "equal_weight_probability_average",
            "checkpoint_epochs": {str(seed): int(self.checkpoints[seed]["epoch"]) for seed in self.seeds},
            "model_loaded_now": raw["model_loaded_now"],
            "model_load_seconds": raw["model_load_seconds"],
            "inference_ms": raw["inference_ms"],
            "input_shape": raw["input_shape"],
            "true_packet_length": raw["true_length"],
            "metadata": metadata,
            "users": users,
        }

    def _prediction_file(self, split: str) -> Path:
        """Return the saved ensemble-prediction file used for verification."""

        if split == "validation":
            return self.settings.path("paths", "ensemble_results_dir") / "validation_predictions.npz"
        if split == "test":
            return self.settings.path("paths", "final_results_dir") / "predictions.npz"
        raise ValueError("Saved ensemble predictions are available only for validation and test.")

    def _ensure_frozen_predictions_loaded(self, split: str) -> dict[str, Any]:
        """Load saved ensemble probabilities for one split once."""

        if split in self.frozen_prediction_cache:
            return self.frozen_prediction_cache[split]
        path = self._prediction_file(split)
        if not path.exists():
            raise FileNotFoundError(path)
        npz = np.load(path)
        payload = {
            "path": path,
            "presence_probabilities": np.asarray(npz["ensemble_presence_probabilities"], dtype=np.float32),
            "activity_probabilities": np.asarray(npz["ensemble_activity_probabilities"], dtype=np.float32),
        }
        self.frozen_prediction_cache[split] = payload
        return payload

    def verify_live_prediction(self, sample_id: str, split: str = "test", atol: float | None = None) -> dict[str, Any]:
        """Check live batch-one inference against saved batch-evaluation probabilities."""

        tolerance = atol if atol is not None else self.settings.getfloat("evaluation", "verification_tolerance")
        row_index, _ = self._row_for_sample(sample_id, split)
        live = self._infer_row(split, row_index)
        frozen = self._ensure_frozen_predictions_loaded(split)
        frozen_presence = frozen["presence_probabilities"][row_index]
        frozen_activity = frozen["activity_probabilities"][row_index]
        max_presence_error = float(np.max(np.abs(live["presence_probabilities"] - frozen_presence)))
        max_activity_error = float(np.max(np.abs(live["activity_probabilities"] - frozen_activity)))
        live_presence = live["presence_probabilities"] >= self.presence_threshold
        frozen_presence_decision = frozen_presence >= self.presence_threshold
        live_activity = np.argmax(live["activity_probabilities"], axis=1)
        frozen_activity_argmax = np.argmax(frozen_activity, axis=1)
        verified = (
            max_presence_error <= tolerance
            and max_activity_error <= tolerance
            and np.array_equal(live_presence, frozen_presence_decision)
            and np.array_equal(live_activity, frozen_activity_argmax)
        )
        return {
            "sample_id": str(sample_id),
            "split": split,
            "row_index": row_index,
            "verified": bool(verified),
            "absolute_tolerance": float(tolerance),
            "max_presence_probability_error": max_presence_error,
            "max_activity_probability_error": max_activity_error,
            "presence_decisions_match": bool(np.array_equal(live_presence, frozen_presence_decision)),
            "activity_argmax_match": bool(np.array_equal(live_activity, frozen_activity_argmax)),
            "inference_ms": float(live["inference_ms"]),
            "frozen_prediction_path": str(frozen["path"]),
        }

    def get_status(self) -> dict[str, Any]:
        """Return application-facing model, device, and artifact status."""

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        final_metrics_path = self.settings.path("paths", "final_results_dir") / "metrics.json"
        final_metrics = None
        if final_metrics_path.exists():
            with open(final_metrics_path, "r", encoding="utf-8") as handle:
                final_metrics = json.load(handle)
        return {
            "model": "Temporal-Pyramid Ensemble",
            "architecture": "Temporal CNN + packed BiLSTM + temporal-pyramid attention",
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu,
            "model_loaded": self.model_loaded,
            "model_load_seconds": self.model_load_seconds,
            "presence_threshold": self.presence_threshold,
            "ensemble": True,
            "ensemble_size": len(self.seeds),
            "ensemble_seeds": self.seeds,
            "ensemble_rule": "equal_weight_probability_average",
            "checkpoint_exists_by_seed": {str(seed): self._checkpoint_path(seed).exists() for seed in self.seeds},
            "checkpoint_epochs": {
                str(seed): int(self.checkpoints[seed]["epoch"]) if seed in self.checkpoints else None
                for seed in self.seeds
            },
            "validation_prediction_file_exists": self._prediction_file("validation").exists(),
            "test_prediction_file_exists": self._prediction_file("test").exists(),
            "final_metrics": final_metrics,
        }
