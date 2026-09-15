"""Read-only access to processed data, frozen predictions, and final metrics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.ml.metrics import activity_confusion_matrix, evaluate_probabilities, unpack_targets
from src.settings import ProjectSettings, load_settings


class WiSenseRepository:
    """Provide application-friendly views of the completed sensing experiment."""

    def __init__(self, settings: ProjectSettings | None = None) -> None:
        """Prepare lazy access to split manifests and final prediction artifacts."""

        self.settings = settings or load_settings()
        self.split_dir = self.settings.path("paths", "split_dir")
        self.cache_dir = self.settings.path("paths", "cache_dir")
        self.final_dir = self.settings.path("paths", "final_results_dir")
        self.ensemble_dir = self.settings.path("paths", "ensemble_results_dir")
        self._frames: dict[str, pd.DataFrame] = {}
        self._final_npz: np.lib.npyio.NpzFile | None = None

    def _read_json(self, path) -> dict[str, Any] | None:
        """Read a JSON file if it exists, otherwise return ``None``."""

        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _frame(self, split: str) -> pd.DataFrame:
        """Load and cache one split manifest."""

        if split not in self._frames:
            path = self.split_dir / f"{split}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            self._frames[split] = pd.read_csv(path, dtype={"sample_id": str})
        return self._frames[split]

    def _final_predictions(self) -> np.lib.npyio.NpzFile:
        """Load the saved final ensemble predictions lazily."""

        if self._final_npz is None:
            path = self.final_dir / "predictions.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            self._final_npz = np.load(path)
        return self._final_npz

    def get_dataset_summary(self) -> dict[str, Any]:
        """Return dataset size, configured labels, environments, and split counts."""

        manifest_path = self.split_dir / "manifest.csv"
        if not manifest_path.exists():
            return {
                "ready": False,
                "message": "Run scripts/01_build_splits.py first.",
                "num_users": self.settings.num_users,
                "num_activities": self.settings.num_activities,
                "activities": self.settings.activities,
            }

        manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
        return {
            "ready": True,
            "total_samples": int(len(manifest)),
            "num_users": self.settings.num_users,
            "num_activities": self.settings.num_activities,
            "activities": self.settings.activities,
            "num_channels": self.settings.getint("dataset", "num_channels"),
            "target_length": self.settings.getint("preprocessing", "target_length"),
            "duration_seconds": self.settings.getfloat("dataset", "duration_seconds"),
            "environments": sorted(manifest["environment"].astype(str).unique().tolist()),
            "wifi_bands_ghz": sorted(manifest["wifi_band_ghz"].astype(float).unique().tolist()),
            "split_counts": {
                split: int(len(self._frame(split)))
                for split in ("train", "validation", "test")
                if (self.split_dir / f"{split}.csv").exists()
            },
        }

    def get_final_model_metrics(self) -> dict[str, Any]:
        """Return final locked test metrics and model-lock metadata when available."""

        metrics = self._read_json(self.final_dir / "metrics.json")
        lock = self._read_json(self.final_dir / "model_lock.json")
        return {
            "ready": metrics is not None,
            "metrics": metrics,
            "lock": lock,
        }

    def get_activity_performance(self, activity: str | None = None):
        """Return saved per-activity final-test precision, recall, F1, and support."""

        path = self.final_dir / "per_class_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if activity is None:
            return json.loads(frame.to_json(orient="records"))
        match = frame[frame["activity"] == activity]
        if match.empty:
            raise ValueError(f"Unknown activity: {activity}")
        return json.loads(match.iloc[0].to_json())

    def get_top_confusions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the largest directional present-user activity confusions."""

        path = self.final_dir / "confusion_matrix.csv"
        if path.exists():
            matrix = pd.read_csv(path, index_col=0)
        else:
            predictions = self._final_predictions()
            matrix = activity_confusion_matrix(
                predictions["flat_targets"],
                predictions["ensemble_activity_probabilities"],
                self.settings.num_users,
                self.settings.activities,
            )

        rows: list[dict[str, Any]] = []
        for true_activity in matrix.index:
            support = int(matrix.loc[true_activity].sum())
            for predicted_activity in matrix.columns:
                if true_activity == predicted_activity:
                    continue
                count = int(matrix.loc[true_activity, predicted_activity])
                rows.append(
                    {
                        "true_activity": str(true_activity),
                        "predicted_activity": str(predicted_activity),
                        "count": count,
                        "true_class_support": support,
                        "rate_given_true_activity": float(count / support) if support else 0.0,
                    }
                )
        rows.sort(key=lambda row: (row["rate_given_true_activity"], row["count"]), reverse=True)
        return rows[: max(1, int(limit))]

    def _group_performance(self, column: str) -> list[dict[str, Any]]:
        """Evaluate final predictions for each unique value of one manifest column."""

        frame = self._frame("test")
        predictions = self._final_predictions()
        targets = np.asarray(predictions["flat_targets"])
        presence = np.asarray(predictions["ensemble_presence_probabilities"])
        activity = np.asarray(predictions["ensemble_activity_probabilities"])
        threshold = self.settings.getfloat("training", "presence_threshold")
        output: list[dict[str, Any]] = []

        for value in sorted(frame[column].unique().tolist()):
            mask = frame[column].to_numpy() == value
            metrics = evaluate_probabilities(
                targets[mask],
                presence[mask],
                activity[mask],
                self.settings.num_users,
                self.settings.num_activities,
                threshold,
            )
            output.append({column: value.item() if isinstance(value, np.generic) else value, **metrics})
        return output

    def get_band_performance(self) -> list[dict[str, Any]]:
        """Return final metrics grouped by WiFi frequency band."""

        return self._group_performance("wifi_band_ghz")

    def get_environment_performance(self) -> list[dict[str, Any]]:
        """Return final metrics grouped by sensing environment."""

        return self._group_performance("environment")

    def get_user_count_performance(self) -> list[dict[str, Any]]:
        """Return final metrics grouped by simultaneous user count."""

        return self._group_performance("number_of_users")

    def get_user_index_performance(self, user_index: int | None = None):
        """Return presence and activity performance for each anonymized user slot."""

        predictions = self._final_predictions()
        targets = np.asarray(predictions["flat_targets"])
        presence_prob = np.asarray(predictions["ensemble_presence_probabilities"])
        activity_prob = np.asarray(predictions["ensemble_activity_probabilities"])
        _, presence_true, activity_true = unpack_targets(
            targets, self.settings.num_users, self.settings.num_activities
        )
        presence_pred = presence_prob >= self.settings.getfloat("training", "presence_threshold")
        activity_pred = np.argmax(activity_prob, axis=2)

        rows: list[dict[str, Any]] = []
        for index in range(self.settings.num_users):
            present_mask = presence_true[:, index] == 1
            activity_f1 = (
                float(
                    f1_score(
                        activity_true[present_mask, index],
                        activity_pred[present_mask, index],
                        labels=list(range(self.settings.num_activities)),
                        average="macro",
                        zero_division=0,
                    )
                )
                if np.any(present_mask)
                else None
            )
            rows.append(
                {
                    "user_index": index + 1,
                    "presence_f1": float(
                        f1_score(presence_true[:, index], presence_pred[:, index], zero_division=0)
                    ),
                    "activity_macro_f1": activity_f1,
                    "present_samples": int(present_mask.sum()),
                }
            )

        if user_index is None:
            return rows
        if not 1 <= int(user_index) <= self.settings.num_users:
            raise ValueError(f"user_index must be between 1 and {self.settings.num_users}.")
        return rows[int(user_index) - 1]

    def get_frozen_prediction(self, row_index: int) -> dict[str, Any]:
        """Return the saved final ensemble probabilities for one test-set row."""

        predictions = self._final_predictions()
        frame = self._frame("test")
        if not 0 <= int(row_index) < len(frame):
            raise IndexError(row_index)
        index = int(row_index)
        return {
            "row_index": index,
            "sample_id": str(frame.iloc[index]["sample_id"]),
            "presence_probabilities": np.asarray(
                predictions["ensemble_presence_probabilities"][index]
            ).astype(float).tolist(),
            "activity_probabilities": np.asarray(
                predictions["ensemble_activity_probabilities"][index]
            ).astype(float).tolist(),
        }

    def get_error_analysis_summary(self) -> dict[str, Any]:
        """Summarize final structured errors without creating another stored artifact."""

        predictions = self._final_predictions()
        targets = np.asarray(predictions["flat_targets"])
        presence_prob = np.asarray(predictions["ensemble_presence_probabilities"])
        activity_prob = np.asarray(predictions["ensemble_activity_probabilities"])
        _, presence_true, activity_true = unpack_targets(
            targets, self.settings.num_users, self.settings.num_activities
        )
        presence_pred = presence_prob >= self.settings.getfloat("training", "presence_threshold")
        activity_pred = np.argmax(activity_prob, axis=2)
        present_mask = presence_true == 1
        activity_errors = present_mask & (activity_true != activity_pred)
        false_presence = (presence_true == 0) & presence_pred
        missed_presence = (presence_true == 1) & (~presence_pred)
        return {
            "activity_errors": int(activity_errors.sum()),
            "false_presence_errors": int(false_presence.sum()),
            "missed_presence_errors": int(missed_presence.sum()),
            "samples_with_any_error": int(
                np.any(activity_errors | false_presence | missed_presence, axis=1).sum()
            ),
            "test_samples": int(targets.shape[0]),
        }
