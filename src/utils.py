"""General utilities used across modules."""
from __future__ import annotations

from pathlib import Path



def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
