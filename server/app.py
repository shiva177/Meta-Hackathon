"""
server/app.py — OpenEnv-required entry point.
Provides a main() function for openenv serve / uv_run deployment modes.
"""

import sys
import os

# Ensure root directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401

import uvicorn


def main() -> None:
    """Start the Customer Support Ticket Resolution environment server."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
