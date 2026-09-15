"""Label utilities for user-specific activity recognition."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def encode_activity_matrix(
    row: pd.Series,
    num_users: int,
    activities: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a user-by-activity target matrix from one annotation row.

    Each present user contributes one one-hot activity label.  Missing user
    activity fields are treated as absent-user slots.
    """

    activity_to_index = {name: index for index, name in enumerate(activities)}
    matrix = np.zeros((num_users, len(activities)), dtype=np.float32)
    presence = np.zeros(num_users, dtype=np.float32)

    for user_index in range(num_users):
        value = row.get(f"user_{user_index + 1}_activity")
        if pd.isna(value):
            continue

        activity = str(value).strip()
        if activity not in activity_to_index:
            raise ValueError(f"Unknown activity label: {activity}")

        matrix[user_index, activity_to_index[activity]] = 1.0
        presence[user_index] = 1.0

    return matrix, presence


def target_columns(num_users: int, activities: Sequence[str]) -> list[str]:
    """Return flattened target column names in model-output order."""

    return [
        f"y_user{user_index}_{activity}"
        for user_index in range(1, num_users + 1)
        for activity in activities
    ]


def presence_columns(num_users: int) -> list[str]:
    """Return user-presence column names used in split manifests."""

    return [f"mask_user_{user_index}" for user_index in range(1, num_users + 1)]
