"""Create reproducible train, validation, and test manifests from WiMANS annotations."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.labels import encode_activity_matrix, presence_columns, target_columns
from src.settings import load_settings


def build_manifest(annotations: pd.DataFrame, settings) -> pd.DataFrame:
    """Convert annotation rows into model labels and stratification metadata."""

    sample_column = settings.get("dataset", "sample_id_column")
    environment_column = settings.get("dataset", "environment_column")
    band_column = settings.get("dataset", "band_column")
    user_count_column = settings.get("dataset", "user_count_column")
    activities = settings.activities
    num_users = settings.num_users

    rows: list[dict] = []
    for _, annotation in annotations.iterrows():
        activity_matrix, presence = encode_activity_matrix(annotation, num_users, activities)
        environment = str(annotation[environment_column])
        band = float(annotation[band_column])
        user_count = int(annotation[user_count_column])

        row = {
            "sample_id": str(annotation[sample_column]),
            "environment": environment,
            "wifi_band_ghz": band,
            "number_of_users": user_count,
            "stratum": f"{environment}|{band:g}|{user_count}",
        }
        for user_index, column in enumerate(presence_columns(num_users)):
            row[column] = int(presence[user_index])
        for column, value in zip(
            target_columns(num_users, activities), activity_matrix.reshape(-1), strict=True
        ):
            row[column] = int(value)
        rows.append(row)

    return pd.DataFrame(rows)


def validate_split_fractions(train_fraction: float, validation_fraction: float, test_fraction: float) -> None:
    """Check that configured split fractions form a valid partition."""

    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0; configured total is {total:.6f}.")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Train, validation, and test fractions must all be positive.")


def main() -> None:
    """Build and save the split manifests."""

    settings = load_settings()
    dataset_dir = settings.path("paths", "dataset_dir")
    annotation_path = dataset_dir / settings.get("dataset", "annotation_file")
    split_dir = settings.path("paths", "split_dir")
    split_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(annotation_path)
    manifest = build_manifest(annotations, settings)

    train_fraction = settings.getfloat("split", "train_fraction")
    validation_fraction = settings.getfloat("split", "validation_fraction")
    test_fraction = settings.getfloat("split", "test_fraction")
    validate_split_fractions(train_fraction, validation_fraction, test_fraction)

    minimum_stratum_size = settings.getint("split", "minimum_stratum_size")
    stratum_counts = manifest["stratum"].value_counts()
    if int(stratum_counts.min()) < minimum_stratum_size:
        raise RuntimeError(
            f"At least one stratum contains fewer than {minimum_stratum_size} samples."
        )

    seed = settings.getint("split", "random_seed")
    train_val, test = train_test_split(
        manifest,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=manifest["stratum"],
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=seed + 1,
        shuffle=True,
        stratify=train_val["stratum"],
    )

    frames = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    sets = {name: set(frame["sample_id"]) for name, frame in frames.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("A sample appears in more than one split.")

    manifest.to_csv(split_dir / "manifest.csv", index=False)
    for name, frame in frames.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)

    summary = {
        "total_samples": int(len(manifest)),
        "train_samples": int(len(frames["train"])),
        "validation_samples": int(len(frames["validation"])),
        "test_samples": int(len(frames["test"])),
        "random_seed": seed,
        "strata": int(len(stratum_counts)),
    }
    with open(split_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print("Split manifests written to", split_dir)
    print(summary)


if __name__ == "__main__":
    main()
