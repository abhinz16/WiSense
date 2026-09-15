"""Start the WiSense browser dashboard and MCP endpoint using ``config.ini``."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from src.settings import load_settings


def main() -> None:
    """Launch Uvicorn with the configured host, port, and keep-alive timeout."""

    settings = load_settings()
    uvicorn.run(
        "src.wisense_mcp.web_app:app",
        host=settings.get("server", "host"),
        port=settings.getint("server", "port"),
        timeout_keep_alive=settings.getint("server", "keep_alive_seconds"),
        reload=False,
    )


if __name__ == "__main__":
    main()
