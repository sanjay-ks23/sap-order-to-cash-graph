"""Vercel serverless entry point — exposes the FastAPI app."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from main import app  # noqa: E402, F401
