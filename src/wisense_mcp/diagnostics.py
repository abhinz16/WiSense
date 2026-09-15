"""CSI integrity checks and retrospective prediction diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.ml.metrics import unpack_targets
from src.settings import ProjectSettings, load_settings


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Return categorical entropy normalized to the interval [0, 1]."""

    values = np.asarray(probabilities, dtype=np.float64)
    values = np.clip(values, 1e-12, 1.0)
    values = values / values.sum()
    entropy = -np.sum(values * np.log(values))
    return float(entropy / math.log(len(values)))


def presence_uncertainty(probability: float, threshold: float) -> float:
    """Return how close a presence probability is to its decision threshold."""

    if probability >= threshold:
        distance = (probability - threshold) / max(1.0 - threshold, 1e-12)
    else:
        distance = (threshold - probability) / max(threshold, 1e-12)
    return float(1.0 - np.clip(distance, 0.0, 1.0))


class WiSenseDiagnostics:
    """Combine cached-signal checks, live inference, and frozen error exploration."""

    def __init__(self, inference_engine, sample_catalog, settings: ProjectSettings | None = None) -> None:
        """Store shared application services and initialize lazy cache handles."""

        self.settings = settings or load_settings()
        self.inference_engine = inference_engine
        self.catalog = sample_catalog
        self.cache_dir = self.settings.path("paths", "cache_dir")
        self.final_dir = self.settings.path("paths", "final_results_dir")
        self._x: dict[str, np.ndarray] = {}
        self._y: dict[str, np.ndarray] = {}
        self._lengths: dict[str, np.ndarray] = {}
        self._final: np.lib.npyio.NpzFile | None = None

    def _ensure_split(self, split: str) -> None:
        """Memory-map cached tensors for one split."""

        if split in self._x:
            return
        self._x[split] = np.load(self.cache_dir / f"{split}_x.npy", mmap_mode="r")
        self._y[split] = np.load(self.cache_dir / f"{split}_y.npy", mmap_mode="r")
        self._lengths[split] = np.load(self.cache_dir / f"{split}_lengths.npy", mmap_mode="r")

    def _final_predictions(self) -> np.lib.npyio.NpzFile:
        """Load final frozen ensemble predictions once."""

        if self._final is None:
            path = self.final_dir / "predictions.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            self._final = np.load(path)
        return self._final

    def analyze_sample(self, identifier: str, split: str = "test") -> dict[str, Any]:
        """Inspect one cached CSI sample and compare live prediction with ground truth."""

        record = self.catalog.resolve(identifier, split)
        self._ensure_split(split)
        row_index = int(record["row_index"])
        sample = np.asarray(self._x[split][row_index], dtype=np.float32)
        target = np.asarray(self._y[split][row_index], dtype=np.float32).reshape(
            self.settings.num_users, self.settings.num_activities
        )
        true_length = int(self._lengths[split][row_index])
        valid = sample[:, :true_length]
        finite_fraction = float(np.isfinite(valid).mean()) if valid.size else 0.0
        temporal_change = (
            float(np.sqrt(np.mean(np.diff(valid, axis=1) ** 2)))
            if valid.shape[1] > 1
            else 0.0
        )
        prediction = self.inference_engine.predict_sample(record["dataset_id"], split, 3)

        users: list[dict[str, Any]] = []
        for user in prediction["users"]:
            user_index = int(user["user_index"]) - 1
            true_present = bool(target[user_index].sum() > 0)
            true_activity = (
                self.settings.activities[int(np.argmax(target[user_index]))] if true_present else None
            )
            predicted_present = bool(user["predicted_present"])
            predicted_activity = user["predicted_activity"]
            structured_correct = true_present == predicted_present and (
                not true_present or true_activity == predicted_activity
            )
            probabilities = np.array(list(user["activity_probabilities"].values()), dtype=float)
            sorted_probabilities = np.sort(probabilities)[::-1]
            users.append(
                {
                    **user,
                    "structured_correct": bool(structured_correct),
                    "activity_metrics_applicable": true_present,
                    "activity_normalized_entropy": normalized_entropy(probabilities) if true_present else None,
                    "activity_top1_top2_margin": (
                        float(sorted_probabilities[0] - sorted_probabilities[1]) if true_present else None
                    ),
                }
            )

        return {
            "sample": record,
            "signal": {
                "packet_coverage": float(min(true_length, self.settings.getint("dataset", "nominal_packets")) / self.settings.getint("dataset", "nominal_packets")),
                "observed_packets": true_length,
                "temporal_change_rms": temporal_change,
                "note": "Signal diagnostics use the cached globally standardized amplitude tensor; they are not RF SNR measurements.",
            },
            "integrity": {
                "finite_fraction": finite_fraction,
                "integrity_status": "good" if finite_fraction == 1.0 else "check input",
            },
            "prediction": {
                "inference_ms": prediction["inference_ms"],
                "device": prediction["device"],
                "activity_metric_note": "Activity uncertainty is reported only for ground-truth present users.",
                "users": users,
            },
        }

    def _error_rows(self) -> list[dict[str, Any]]:
        """Build per-sample error records from the frozen final predictions."""

        predictions = self._final_predictions()
        targets = np.asarray(predictions["flat_targets"])
        presence_prob = np.asarray(predictions["ensemble_presence_probabilities"])
        activity_prob = np.asarray(predictions["ensemble_activity_probabilities"])
        _, presence_true, activity_true = unpack_targets(
            targets, self.settings.num_users, self.settings.num_activities
        )
        threshold = self.settings.getfloat("training", "presence_threshold")
        presence_pred = presence_prob >= threshold
        activity_pred = np.argmax(activity_prob, axis=2)
        rows: list[dict[str, Any]] = []

        for row_index in range(targets.shape[0]):
            present = presence_true[row_index] == 1
            activity_error = present & (activity_true[row_index] != activity_pred[row_index])
            false_presence = (presence_true[row_index] == 0) & presence_pred[row_index]
            missed_presence = (presence_true[row_index] == 1) & (~presence_pred[row_index])
            structured = activity_error | false_presence | missed_presence
            rows.append(
                {
                    "row_index": row_index,
                    "activity_error_count": int(activity_error.sum()),
                    "false_presence_count": int(false_presence.sum()),
                    "missed_presence_count": int(missed_presence.sum()),
                    "structured_error_count": int(structured.sum()),
                    "presence_probabilities": presence_prob[row_index],
                    "activity_probabilities": activity_prob[row_index],
                    "presence_true": presence_true[row_index],
                }
            )
        return rows

    def find_model_errors(
        self,
        error_type: str = "any",
        limit: int = 30,
        environment: str | None = None,
        band: float | None = None,
        user_count: int | None = None,
    ) -> dict[str, Any]:
        """Search frozen final predictions for selected error types and metadata filters."""

        valid_types = {"any", "activity", "false_presence", "missed_presence", "structured"}
        if error_type not in valid_types:
            raise ValueError(f"error_type must be one of {sorted(valid_types)}")

        matches: list[dict[str, Any]] = []
        for row in self._error_rows():
            record = self.catalog.records["test"][row["row_index"]]
            if environment and record["environment"] != environment:
                continue
            if band is not None and float(record["wifi_band_ghz"]) != float(band):
                continue
            if user_count is not None and int(record["number_of_users"]) != int(user_count):
                continue

            include = {
                "any": row["structured_error_count"] > 0,
                "structured": row["structured_error_count"] > 0,
                "activity": row["activity_error_count"] > 0,
                "false_presence": row["false_presence_count"] > 0,
                "missed_presence": row["missed_presence_count"] > 0,
            }[error_type]
            if include:
                matches.append({"sample": dict(record), **{key: value for key, value in row.items() if not isinstance(value, np.ndarray)}})

        return {
            "error_type": error_type,
            "matching_samples": len(matches),
            "returned_samples": min(len(matches), max(1, int(limit))),
            "samples": matches[: max(1, int(limit))],
        }

    def find_difficult_samples(
        self,
        limit: int = 30,
        environment: str | None = None,
        band: float | None = None,
        user_count: int | None = None,
    ) -> dict[str, Any]:
        """Rank frozen test samples with a retrospective uncertainty heuristic."""

        threshold = self.settings.getfloat("training", "presence_threshold")
        ranked: list[dict[str, Any]] = []
        for row in self._error_rows():
            record = self.catalog.records["test"][row["row_index"]]
            if environment and record["environment"] != environment:
                continue
            if band is not None and float(record["wifi_band_ghz"]) != float(band):
                continue
            if user_count is not None and int(record["number_of_users"]) != int(user_count):
                continue

            present_mask = row["presence_true"] == 1
            entropies = [
                normalized_entropy(row["activity_probabilities"][index])
                for index in np.where(present_mask)[0]
            ]
            margins = []
            for index in np.where(present_mask)[0]:
                probabilities = np.sort(row["activity_probabilities"][index])[::-1]
                margins.append(float(probabilities[0] - probabilities[1]))
            presence_scores = [
                presence_uncertainty(float(probability), threshold)
                for probability in row["presence_probabilities"]
            ]
            mean_entropy = float(np.mean(entropies)) if entropies else 0.0
            mean_margin = float(np.mean(margins)) if margins else 1.0
            mean_presence_uncertainty = float(np.mean(presence_scores))
            difficulty = (
                self.settings.getfloat("diagnostics", "activity_entropy_weight") * mean_entropy
                + self.settings.getfloat("diagnostics", "activity_margin_weight") * (1.0 - mean_margin)
                + self.settings.getfloat("diagnostics", "presence_uncertainty_weight") * mean_presence_uncertainty
            )
            ranked.append(
                {
                    "sample": dict(record),
                    "difficulty_score": float(difficulty),
                    "mean_activity_entropy": mean_entropy,
                    "mean_activity_top1_top2_margin": mean_margin,
                    "mean_presence_uncertainty": mean_presence_uncertainty,
                    "structured_error_count": row["structured_error_count"],
                }
            )

        ranked.sort(key=lambda item: item["difficulty_score"], reverse=True)
        return {
            "returned_samples": min(len(ranked), max(1, int(limit))),
            "samples": ranked[: max(1, int(limit))],
            "note": "Difficulty is a retrospective heuristic, not calibrated uncertainty.",
        }
