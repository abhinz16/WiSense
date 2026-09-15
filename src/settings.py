"""Project configuration helpers.

Every user-editable setting lives in ``config.ini``.  Code should read values
through this module instead of embedding machine-specific paths or experiment
settings in individual scripts.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.ini"


@dataclass(frozen=True)
class ProjectSettings:
    """Thin wrapper around ``ConfigParser`` with project-aware path handling."""

    parser: ConfigParser
    config_path: Path
    project_root: Path

    def get(self, section: str, option: str) -> str:
        """Return a string value from the configuration file."""

        return self.parser.get(section, option)

    def getint(self, section: str, option: str) -> int:
        """Return an integer value from the configuration file."""

        return self.parser.getint(section, option)

    def getfloat(self, section: str, option: str) -> float:
        """Return a floating-point value from the configuration file."""

        return self.parser.getfloat(section, option)

    def getboolean(self, section: str, option: str) -> bool:
        """Return a Boolean value from the configuration file."""

        return self.parser.getboolean(section, option)

    def get_list(self, section: str, option: str) -> list[str]:
        """Return a comma-separated configuration value as a cleaned list."""

        raw_value = self.get(section, option)
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def get_int_list(self, section: str, option: str) -> list[int]:
        """Return a comma-separated configuration value as integers."""

        return [int(item) for item in self.get_list(section, option)]

    def path(self, section: str, option: str) -> Path:
        """Resolve a configured path relative to the repository root."""

        configured = Path(self.get(section, option)).expanduser()
        if configured.is_absolute():
            return configured
        return (self.project_root / configured).resolve()

    @property
    def activities(self) -> list[str]:
        """Return the activity labels in model-output order."""

        return self.get_list("dataset", "activities")

    @property
    def num_users(self) -> int:
        """Return the number of anonymized user slots in the dataset."""

        return self.getint("dataset", "num_users")

    @property
    def num_activities(self) -> int:
        """Return the number of configured activity classes."""

        return len(self.activities)


def load_settings(config_path: str | Path | None = None) -> ProjectSettings:
    """Load ``config.ini`` and return a validated settings object.

    Parameters
    ----------
    config_path:
        Optional alternate configuration file.  Normal project use relies on
        the repository-root ``config.ini``.
    """

    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    parser = ConfigParser()
    parser.read(path)

    required_sections: Iterable[str] = (
        "project",
        "paths",
        "dataset",
        "split",
        "preprocessing",
        "model",
        "training",
        "diagnostics",
        "evaluation",
        "server",
    )
    missing = [section for section in required_sections if section not in parser]
    if missing:
        raise ValueError(f"config.ini is missing sections: {', '.join(missing)}")

    return ProjectSettings(parser=parser, config_path=path, project_root=PROJECT_ROOT)
