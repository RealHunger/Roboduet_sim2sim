"""Add the RoboDuet repo root to sys.path for direct utility execution."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root
