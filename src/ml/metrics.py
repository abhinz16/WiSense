"""Metrics shared by validation, test evaluation, and application analytics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score


def unpack_targets(flat_targets: np.ndarray, num_users: int, num_activities: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target matrix, presence labels, and activity indices."""

    matrix = np.asarray(flat_targets).reshape(-1, num_users, num_activities)
    presence = (matrix.sum(axis=2) > 0).astype(np.int32)
    activity = np.argmax(matrix, axis=2)
    return matrix, presence, activity


def evaluate_probabilities(
    flat_targets: np.ndarray,
    presence_probabilities: np.ndarray,
    activity_probabilities: np.ndarray,
    num_users: int,
    num_activities: int,
    presence_threshold: float,
) -> dict[str, float]:
    """Calculate the project's frozen presence, activity, and structured metrics."""

    _, presence_true, activity_true = unpack_targets(flat_targets, num_users, num_activities)
    presence_pred = (presence_probabilities >= presence_threshold).astype(np.int32)
    present_mask = presence_true == 1
    activity_pred = np.argmax(activity_probabilities, axis=2)

    true_present = activity_true[present_mask]
    pred_present = activity_pred[present_mask]
    absent_mask = presence_true == 0

    true_state = np.zeros_like(activity_true, dtype=np.int32)
    true_state[present_mask] = activity_true[present_mask] + 1
    pred_state = np.zeros_like(activity_pred, dtype=np.int32)
    predicted_present_mask = presence_pred == 1
    pred_state[predicted_present_mask] = activity_pred[predicted_present_mask] + 1

    return {
        "presence_macro_f1": float(f1_score(presence_true, presence_pred, average="macro", zero_division=0)),
        "presence_precision": float(precision_score(presence_true, presence_pred, average="macro", zero_division=0)),
        "presence_recall": float(recall_score(presence_true, presence_pred, average="macro", zero_division=0)),
        "activity_top1_accuracy": float(accuracy_score(true_present, pred_present)),
        "activity_macro_f1": float(
            f1_score(
                true_present,
                pred_present,
                labels=list(range(num_activities)),
                average="macro",
                zero_division=0,
            )
        ),
        "structured_macro_f1": float(
            f1_score(
                true_state.reshape(-1),
                pred_state.reshape(-1),
                labels=list(range(num_activities + 1)),
                average="macro",
                zero_division=0,
            )
        ),
        "structured_accuracy": float(accuracy_score(true_state.reshape(-1), pred_state.reshape(-1))),
        "absent_user_fpr": float(presence_pred[absent_mask].mean()) if np.any(absent_mask) else 0.0,
    }


def per_class_activity_metrics(
    flat_targets: np.ndarray,
    activity_probabilities: np.ndarray,
    num_users: int,
    activities: Sequence[str],
) -> pd.DataFrame:
    """Return precision, recall, F1, and support for each present-user activity."""

    num_activities = len(activities)
    _, presence_true, activity_true = unpack_targets(flat_targets, num_users, num_activities)
    activity_pred = np.argmax(activity_probabilities, axis=2)
    mask = presence_true == 1
    precision, recall, f1, support = precision_recall_fscore_support(
        activity_true[mask],
        activity_pred[mask],
        labels=list(range(num_activities)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "activity": list(activities),
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def activity_confusion_matrix(
    flat_targets: np.ndarray,
    activity_probabilities: np.ndarray,
    num_users: int,
    activities: Sequence[str],
) -> pd.DataFrame:
    """Return the present-user activity confusion matrix as a labeled DataFrame."""

    num_activities = len(activities)
    _, presence_true, activity_true = unpack_targets(flat_targets, num_users, num_activities)
    activity_pred = np.argmax(activity_probabilities, axis=2)
    mask = presence_true == 1
    matrix = confusion_matrix(
        activity_true[mask], activity_pred[mask], labels=list(range(num_activities))
    )
    return pd.DataFrame(matrix, index=list(activities), columns=list(activities))
