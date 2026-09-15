"""Human-readable sample aliases for browser and MCP interactions."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.settings import ProjectSettings, load_settings


def _slugify(value: str) -> str:
    """Convert a short label into a URL-friendly token."""

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _environment_display(value: str) -> str:
    """Convert an environment identifier into a readable label."""

    return value.replace("_", " ").title()


def _people_text(count: int) -> str:
    """Return a grammatically correct user-count label."""

    if count == 0:
        return "No people"
    if count == 1:
        return "1 person"
    return f"{count} people"


class SampleCatalog:
    """Index split manifests and expose stable aliases instead of raw dataset IDs."""

    def __init__(self, settings: ProjectSettings | None = None) -> None:
        """Load split metadata from the configured processed-data directory."""

        self.settings = settings or load_settings()
        self.split_dir = self.settings.path("paths", "split_dir")
        self.frames: dict[str, pd.DataFrame] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.alias_lookup: dict[str, dict[str, dict[str, Any]]] = {}
        self.raw_lookup: dict[str, dict[str, dict[str, Any]]] = {}
        self._load_splits()

    def _load_splits(self) -> None:
        """Read available split CSV files and build lookup tables."""

        for split in ("train", "validation", "test"):
            path = self.split_dir / f"{split}.csv"
            if not path.exists():
                continue

            frame = pd.read_csv(path, dtype={"sample_id": str})
            records: list[dict[str, Any]] = []
            alias_lookup: dict[str, dict[str, Any]] = {}
            raw_lookup: dict[str, dict[str, Any]] = {}

            for row_index, row in frame.iterrows():
                raw_id = str(row["sample_id"])
                environment = str(row["environment"])
                band = float(row["wifi_band_ghz"])
                user_count = int(row["number_of_users"])
                alias = (
                    f"{_slugify(environment)}-"
                    f"{str(band).replace('.', 'g')}-"
                    f"{user_count}p-{row_index + 1:04d}"
                )
                display_name = (
                    f"{_environment_display(environment)} · {band:g} GHz · "
                    f"{_people_text(user_count)} · Sample {row_index + 1:04d}"
                )
                record = {
                    "alias": alias,
                    "display_name": display_name,
                    "environment": environment,
                    "environment_display": _environment_display(environment),
                    "wifi_band_ghz": band,
                    "number_of_users": user_count,
                    "split": split,
                    "row_index": int(row_index),
                    "dataset_id": raw_id,
                }
                records.append(record)
                alias_lookup[alias] = record
                raw_lookup[raw_id] = record

            self.frames[split] = frame
            self.records[split] = records
            self.alias_lookup[split] = alias_lookup
            self.raw_lookup[split] = raw_lookup

    def resolve(self, identifier: str, split: str = "test") -> dict[str, Any]:
        """Resolve either a human-readable alias or a raw sample ID."""

        split = split.lower().strip()
        if split not in self.records:
            raise ValueError(f"Split '{split}' is unavailable.")
        key = str(identifier)
        if key in self.alias_lookup[split]:
            return dict(self.alias_lookup[split][key])
        if key in self.raw_lookup[split]:
            return dict(self.raw_lookup[split][key])
        raise ValueError(f"Sample '{identifier}' was not found in split '{split}'.")

    def list_samples(
        self,
        split: str = "test",
        limit: int = 50,
        offset: int = 0,
        environment: str | None = None,
        band: float | None = None,
        user_count: int | None = None,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Return filtered sample records for the dashboard or MCP client."""

        split = split.lower().strip()
        if split not in self.records:
            raise ValueError(f"Split '{split}' is unavailable.")

        records = self.records[split]
        filtered: list[dict[str, Any]] = []
        search_text = search.lower().strip() if search else None

        for record in records:
            if environment and record["environment"] != environment:
                continue
            if band is not None and float(record["wifi_band_ghz"]) != float(band):
                continue
            if user_count is not None and int(record["number_of_users"]) != int(user_count):
                continue
            if search_text:
                haystack = " ".join(
                    [record["alias"], record["display_name"], record["dataset_id"]]
                ).lower()
                if search_text not in haystack:
                    continue
            filtered.append(record)

        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 500))
        return {
            "split": split,
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "samples": [dict(record) for record in filtered[offset : offset + limit]],
        }

    def filters(self, split: str = "test") -> dict[str, list[Any]]:
        """Return the distinct filter choices available in one split."""

        split = split.lower().strip()
        if split not in self.frames:
            raise ValueError(f"Split '{split}' is unavailable.")
        frame = self.frames[split]
        return {
            "environments": sorted(frame["environment"].astype(str).unique().tolist()),
            "bands": sorted(frame["wifi_band_ghz"].astype(float).unique().tolist()),
            "user_counts": sorted(frame["number_of_users"].astype(int).unique().tolist()),
        }
