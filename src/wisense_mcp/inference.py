"""Production inference for cached WiMANS samples and uploaded CSI measurements."""

from __future__ import annotations

import io
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
    """Load the trained ensemble on demand and serve CSI predictions."""

    def __init__(self, settings: ProjectSettings | None = None) -> None:
        """Prepare lazy model, cache, normalization, and verification state."""

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
        self._normalization_mean: np.ndarray | None = None
        self._normalization_std: np.ndarray | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _checkpoint_path(self, seed: int) -> Path:
        """Return the configured checkpoint path for one ensemble seed."""

        return self.model_dir / f"model_seed_{seed}.pt"

    def _ensure_models_loaded(self) -> bool:
        """Load all ensemble checkpoints once and report whether loading happened now."""

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

    def _ensure_normalization_loaded(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the training-derived channel statistics used by every prediction."""

        if self._normalization_mean is not None and self._normalization_std is not None:
            return self._normalization_mean, self._normalization_std

        path = self.settings.path("paths", "normalization_file")
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)

        expected_channels = self.settings.getint("dataset", "num_channels")
        if mean.shape != (expected_channels,) or std.shape != (expected_channels,):
            raise ValueError(
                "Normalization statistics do not match the configured CSI channel count: "
                f"mean={mean.shape}, std={std.shape}, expected=({expected_channels},)."
            )
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Normalization statistics contain invalid values.")

        self._normalization_mean = mean
        self._normalization_std = std
        return mean, std

    def _ensure_split_loaded(self, split: str) -> dict[str, Any]:
        """Memory-map one processed split and its metadata table."""

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
            "sample_to_row": {
                sample_id: index
                for index, sample_id in enumerate(metadata["sample_id"].astype(str))
            },
        }

        if not (
            len(payload["x"])
            == len(payload["y"])
            == len(payload["lengths"])
            == len(metadata)
        ):
            raise RuntimeError(f"Processed split '{split}' is not row-aligned.")

        self.split_cache[split] = payload
        return payload

    def _row_for_sample(self, sample_id: str, split: str) -> tuple[int, pd.Series]:
        """Return the processed row index and metadata for a dataset sample ID."""

        data = self._ensure_split_loaded(split)
        key = str(sample_id)
        if key not in data["sample_to_row"]:
            raise KeyError(f"Sample '{sample_id}' was not found in split '{split}'.")

        row_index = int(data["sample_to_row"][key])
        return row_index, data["metadata"].iloc[row_index]

    @torch.inference_mode()
    def _infer_tensor(self, sample: np.ndarray, length: int) -> dict[str, Any]:
        """Run the ensemble on one preprocessed ``[channels, time]`` tensor."""

        if sample.ndim != 2:
            raise ValueError(f"Expected one CSI tensor with shape [C, T], got {sample.shape}.")

        expected_channels = self.settings.getint("dataset", "num_channels")
        if sample.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} CSI channels, got {sample.shape[0]}."
            )

        loaded_now = self._ensure_models_loaded()
        valid_length = max(1, min(int(length), sample.shape[1]))
        x = torch.from_numpy(np.asarray(sample, dtype=np.float32).copy()).unsqueeze(0)
        x = x.to(self.device, dtype=torch.float32)
        lengths = torch.tensor([valid_length], device=self.device, dtype=torch.long)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        presence_runs: list[np.ndarray] = []
        activity_runs: list[np.ndarray] = []

        with self._inference_lock:
            for seed in self.seeds:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    presence_logits, activity_logits = self.models[seed](x, lengths)

                presence_runs.append(torch.sigmoid(presence_logits).float().cpu().numpy()[0])
                activity_runs.append(
                    torch.softmax(activity_logits, dim=2).float().cpu().numpy()[0]
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = float((time.perf_counter() - start) * 1000.0)
        presence_stack = np.stack(presence_runs, axis=0)
        activity_stack = np.stack(activity_runs, axis=0)

        return {
            "presence_probabilities": presence_stack.mean(
                axis=0, dtype=np.float64
            ).astype(np.float32),
            "activity_probabilities": activity_stack.mean(
                axis=0, dtype=np.float64
            ).astype(np.float32),
            "per_model_presence_probabilities": presence_stack,
            "per_model_activity_probabilities": activity_stack,
            "true_length": valid_length,
            "input_shape": list(x.shape),
            "model_loaded_now": loaded_now,
            "model_load_seconds": self.model_load_seconds,
            "inference_ms": inference_ms,
        }

    @torch.inference_mode()
    def _infer_row(self, split: str, row_index: int) -> dict[str, Any]:
        """Run the ensemble for one sample already stored in the tensor cache."""

        data = self._ensure_split_loaded(split)
        sample = np.asarray(data["x"][row_index], dtype=np.float32)
        length = max(1, min(int(data["lengths"][row_index]), sample.shape[1]))
        return self._infer_tensor(sample, length)

    def _format_users(
        self,
        raw: dict[str, Any],
        top_k: int,
        true_presence: np.ndarray | None = None,
        true_activity: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Convert raw model probabilities into user-facing prediction records."""

        top_k = max(1, min(int(top_k), self.settings.num_activities))
        users: list[dict[str, Any]] = []

        for user_index in range(self.settings.num_users):
            presence_probability = float(raw["presence_probabilities"][user_index])
            predicted_present = presence_probability >= self.presence_threshold
            probabilities = raw["activity_probabilities"][user_index]
            activity_index = int(np.argmax(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            actual_present = (
                bool(true_presence[user_index]) if true_presence is not None else None
            )
            actual_activity_index = (
                int(true_activity[user_index])
                if true_activity is not None and actual_present
                else None
            )

            users.append(
                {
                    "user_index": user_index + 1,
                    "predicted_present": bool(predicted_present),
                    "presence_probability": presence_probability,
                    "activity_argmax": activity_index,
                    "activity_argmax_probability": float(probabilities[activity_index]),
                    "predicted_activity": (
                        self.settings.activities[activity_index]
                        if predicted_present
                        else None
                    ),
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
                        for name, probability in zip(
                            self.settings.activities, probabilities, strict=True
                        )
                    },
                    "true_present": actual_present,
                    "true_activity": (
                        self.settings.activities[actual_activity_index]
                        if actual_activity_index is not None
                        else None
                    ),
                    "true_activity_index": actual_activity_index,
                }
            )

        return users

    def predict_sample(
        self, sample_id: str, split: str = "test", top_k: int = 3
    ) -> dict[str, Any]:
        """Return ensemble predictions and ground truth for one dataset sample."""

        split = split.lower().strip()
        row_index, metadata_row = self._row_for_sample(sample_id, split)
        raw = self._infer_row(split, row_index)
        data = self._ensure_split_loaded(split)
        target = np.asarray(data["y"][row_index], dtype=np.float32).reshape(
            self.settings.num_users, self.settings.num_activities
        )
        true_presence = target.sum(axis=1) > 0
        true_activity = np.argmax(target, axis=1)
        users = self._format_users(raw, top_k, true_presence, true_activity)
        metadata = {str(key): _json_safe(value) for key, value in metadata_row.items()}

        return {
            "sample_id": str(sample_id),
            "split": split,
            "row_index": row_index,
            "source": "dataset",
            "model": "Temporal-Pyramid Ensemble",
            "device": str(self.device),
            "presence_threshold": self.presence_threshold,
            "ensemble_size": len(self.seeds),
            "ensemble_seeds": self.seeds,
            "ensemble_rule": "equal_weight_probability_average",
            "checkpoint_epochs": {
                str(seed): int(self.checkpoints[seed]["epoch"]) for seed in self.seeds
            },
            "model_loaded_now": raw["model_loaded_now"],
            "model_load_seconds": raw["model_load_seconds"],
            "inference_ms": raw["inference_ms"],
            "input_shape": raw["input_shape"],
            "true_packet_length": raw["true_length"],
            "metadata": metadata,
            "users": users,
        }

    def _decode_uploaded_array(
        self, payload: bytes, filename: str
    ) -> tuple[np.ndarray, str | None]:
        """Decode a supported uploaded file into a NumPy array."""

        suffix = Path(filename).suffix.lower()
        buffer = io.BytesIO(payload)

        if suffix == ".npy":
            return np.asarray(np.load(buffer, allow_pickle=False)), None

        if suffix == ".npz":
            with np.load(buffer, allow_pickle=False) as archive:
                preferred_keys = ("csi", "amplitude", "amp", "data", "sample")
                selected_key = next(
                    (key for key in preferred_keys if key in archive.files),
                    None,
                )
                if selected_key is None:
                    numeric_keys = [
                        key
                        for key in archive.files
                        if np.asarray(archive[key]).dtype.kind in "biufc"
                    ]
                    if not numeric_keys:
                        raise ValueError(
                            "The NPZ file does not contain a numeric CSI array."
                        )
                    selected_key = numeric_keys[0]

                return np.asarray(archive[selected_key]), selected_key

        if suffix in {".csv", ".txt"}:
            try:
                text = payload.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "Text CSI files must use UTF-8 encoding."
                ) from error

            delimiter = "," if suffix == ".csv" else None
            try:
                array = np.genfromtxt(
                    io.StringIO(text),
                    delimiter=delimiter,
                    dtype=np.float64,
                    invalid_raise=False,
                )
                if np.size(array) == 0 or np.all(~np.isfinite(array)):
                    raise ValueError
            except (ValueError, TypeError):
                array = np.genfromtxt(
                    io.StringIO(text),
                    delimiter=delimiter,
                    dtype=np.complex128,
                    invalid_raise=False,
                )

            array = np.asarray(array)
            if array.ndim == 2:
                keep_rows = ~np.all(~np.isfinite(array), axis=1)
                array = array[keep_rows]
                if array.size and array.ndim == 2:
                    keep_columns = ~np.all(~np.isfinite(array), axis=0)
                    array = array[:, keep_columns]

            return array, None

        raise ValueError(
            f"Unsupported sample format '{suffix or 'unknown'}'. "
            "Use one of the formats configured in config.ini."
        )

    def _coerce_packet_channel_matrix(
        self, array: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Convert common CSI array layouts into ``[packets, channels]`` form."""

        values = np.asarray(array)
        if values.ndim == 0 or values.size == 0:
            raise ValueError("The uploaded sample is empty.")

        channels = self.settings.getint("dataset", "num_channels")
        original_shape = tuple(int(size) for size in values.shape)

        if values.ndim == 1:
            if values.size % channels != 0:
                raise ValueError(
                    f"A flat sample must contain a multiple of {channels} values; "
                    f"received {values.size}."
                )
            matrix = values.reshape(-1, channels)
            layout = "flat values reshaped to packets × channels"

        elif values.ndim == 2:
            if values.shape[1] == channels:
                matrix = values
                layout = "packets × channels"
            elif values.shape[0] == channels:
                matrix = values.T
                layout = "channels × packets (transposed)"
            else:
                raise ValueError(
                    f"Expected one dimension to contain {channels} CSI channels; "
                    f"received shape {original_shape}."
                )

        else:
            packet_axes = [
                axis
                for axis in range(values.ndim)
                if int(np.prod([size for i, size in enumerate(values.shape) if i != axis]))
                == channels
            ]
            if len(packet_axes) != 1:
                raise ValueError(
                    "Could not determine the packet axis automatically. For multi-dimensional "
                    f"CSI, all non-packet dimensions must multiply to {channels}; "
                    f"received shape {original_shape}."
                )

            packet_axis = packet_axes[0]
            moved = np.moveaxis(values, packet_axis, 0)
            matrix = moved.reshape(moved.shape[0], channels)
            layout = (
                f"multi-dimensional CSI with axis {packet_axis} interpreted as packets"
            )

        return np.asarray(matrix), layout

    def _preprocess_external_array(
        self, array: np.ndarray
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Apply the same amplitude normalization and padding used during training."""

        original_shape = [int(size) for size in np.asarray(array).shape]
        matrix, detected_layout = self._coerce_packet_channel_matrix(array)
        was_complex = bool(np.iscomplexobj(matrix))

        if was_complex:
            amplitude = np.abs(matrix).astype(np.float32)
            representation = "complex CSI converted to amplitude"
        else:
            amplitude = np.asarray(matrix, dtype=np.float32)
            representation = "amplitude"

        packet_count = int(amplitude.shape[0])
        if packet_count < 1:
            raise ValueError("The uploaded sample does not contain any CSI packets.")

        mean, std = self._ensure_normalization_loaded()
        finite = np.isfinite(amplitude)
        repaired_values = int(amplitude.size - np.count_nonzero(finite))
        if repaired_values:
            amplitude = np.where(finite, amplitude, mean[None, :])

        normalized = (amplitude - mean[None, :]) / std[None, :]
        target_length = self.settings.getint("preprocessing", "target_length")
        num_channels = self.settings.getint("dataset", "num_channels")
        valid_length = min(packet_count, target_length)
        tensor = np.zeros((num_channels, target_length), dtype=np.float32)
        tensor[:, :valid_length] = normalized[:valid_length].T

        details = {
            "original_shape": original_shape,
            "detected_layout": detected_layout,
            "representation": representation,
            "packet_count": packet_count,
            "packets_used": valid_length,
            "packets_truncated": max(0, packet_count - target_length),
            "padding_packets": max(0, target_length - valid_length),
            "num_channels": num_channels,
            "target_length": target_length,
            "nonfinite_values_repaired": repaired_values,
            "normalization": "training_global_per_channel_z_score",
        }
        return tensor, valid_length, details

    def _signal_preview(
        self, tensor: np.ndarray, length: int, points: int | None = None
    ) -> dict[str, Any]:
        """Create a compact normalized CSI-energy trace for the browser."""

        valid = np.asarray(tensor[:, : max(1, length)], dtype=np.float32)
        energy = np.sqrt(np.mean(valid * valid, axis=0))
        requested_points = points or self.settings.getint("server", "signal_preview_points")
        requested_points = max(1, min(int(requested_points), 500, len(energy)))
        downsampled = np.array(
            [float(group.mean()) for group in np.array_split(energy, requested_points)],
            dtype=np.float32,
        )

        lower, upper = np.percentile(downsampled, [5, 95])
        if upper > lower:
            normalized = np.clip((downsampled - lower) / (upper - lower), 0.0, 1.0)
        else:
            normalized = np.zeros_like(downsampled)

        target_length = self.settings.getint("preprocessing", "target_length")
        nominal_duration = self.settings.getfloat("dataset", "duration_seconds")
        duration = nominal_duration * (max(1, length) / target_length)
        time_seconds = np.linspace(0.0, duration, len(normalized))

        return {
            "signal_type": "RMS across globally standardized CSI-amplitude channels",
            "duration_seconds": float(duration),
            "points": [
                {"t": float(time_value), "value": float(value)}
                for time_value, value in zip(time_seconds, normalized, strict=True)
            ],
        }

    def predict_uploaded_sample(
        self,
        payload: bytes,
        filename: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Predict an uploaded sensor sample without writing the file to disk."""

        if not payload:
            raise ValueError("The uploaded sample is empty.")

        array, array_key = self._decode_uploaded_array(payload, filename)
        tensor, valid_length, preprocessing = self._preprocess_external_array(array)
        raw = self._infer_tensor(tensor, valid_length)
        users = self._format_users(raw, top_k)
        preview = self._signal_preview(tensor, valid_length)

        return {
            "sample_id": filename,
            "split": None,
            "row_index": None,
            "source": "uploaded_sensor_sample",
            "sample": {
                "alias": None,
                "dataset_id": None,
                "display_name": filename,
                "source": "Uploaded sensor sample",
            },
            "model": "Temporal-Pyramid Ensemble",
            "device": str(self.device),
            "presence_threshold": self.presence_threshold,
            "ensemble_size": len(self.seeds),
            "ensemble_seeds": self.seeds,
            "ensemble_rule": "equal_weight_probability_average",
            "checkpoint_epochs": {
                str(seed): int(self.checkpoints[seed]["epoch"]) for seed in self.seeds
            },
            "model_loaded_now": raw["model_loaded_now"],
            "model_load_seconds": raw["model_load_seconds"],
            "inference_ms": raw["inference_ms"],
            "input_shape": raw["input_shape"],
            "true_packet_length": raw["true_length"],
            "metadata": {
                "filename": filename,
                "npz_array_key": array_key,
                "ground_truth_available": False,
            },
            "preprocessing": preprocessing,
            "signal": preview,
            "users": users,
            "compatibility_note": (
                "Predictions assume the uploaded sensor provides the same 270 CSI channel "
                "meaning and antenna/subcarrier ordering used for model training."
            ),
        }

    def _prediction_file(self, split: str) -> Path:
        """Return the saved ensemble-prediction file used for verification."""

        if split == "validation":
            return self.settings.path("paths", "ensemble_results_dir") / "validation_predictions.npz"
        if split == "test":
            return self.settings.path("paths", "final_results_dir") / "predictions.npz"
        raise ValueError(
            "Saved ensemble predictions are available only for validation and test."
        )

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
            "presence_probabilities": np.asarray(
                npz["ensemble_presence_probabilities"], dtype=np.float32
            ),
            "activity_probabilities": np.asarray(
                npz["ensemble_activity_probabilities"], dtype=np.float32
            ),
        }
        self.frozen_prediction_cache[split] = payload
        return payload

    def verify_live_prediction(
        self, sample_id: str, split: str = "test", atol: float | None = None
    ) -> dict[str, Any]:
        """Check live batch-one inference against saved batch-evaluation probabilities."""

        tolerance = (
            atol
            if atol is not None
            else self.settings.getfloat("evaluation", "verification_tolerance")
        )
        row_index, _ = self._row_for_sample(sample_id, split)
        live = self._infer_row(split, row_index)
        frozen = self._ensure_frozen_predictions_loaded(split)
        frozen_presence = frozen["presence_probabilities"][row_index]
        frozen_activity = frozen["activity_probabilities"][row_index]
        max_presence_error = float(
            np.max(np.abs(live["presence_probabilities"] - frozen_presence))
        )
        max_activity_error = float(
            np.max(np.abs(live["activity_probabilities"] - frozen_activity))
        )
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
            "presence_decisions_match": bool(
                np.array_equal(live_presence, frozen_presence_decision)
            ),
            "activity_argmax_match": bool(
                np.array_equal(live_activity, frozen_activity_argmax)
            ),
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
            "checkpoint_exists_by_seed": {
                str(seed): self._checkpoint_path(seed).exists() for seed in self.seeds
            },
            "checkpoint_epochs": {
                str(seed): (
                    int(self.checkpoints[seed]["epoch"])
                    if seed in self.checkpoints
                    else None
                )
                for seed in self.seeds
            },
            "validation_prediction_file_exists": self._prediction_file(
                "validation"
            ).exists(),
            "test_prediction_file_exists": self._prediction_file("test").exists(),
            "final_metrics": final_metrics,
        }
