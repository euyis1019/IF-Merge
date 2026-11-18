"""Where data lives."""

from __future__ import annotations

import os
from pathlib import Path


def default_data_root() -> Path:
    """The data root: ``$IF_MERGE_DATA``, or ``./data``."""
    return Path(os.environ.get("IF_MERGE_DATA", "data"))
