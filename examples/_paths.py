"""Utilities for configuring import paths in example scripts."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"


def add_src_to_syspath() -> None:
    """Ensure the repository's ``src`` directory is on ``sys.path``.

    This allows running the example scripts directly via ``python examples/foo.py``
    without needing to install the package first.
    """

    src_str = str(SRC_PATH)
    if SRC_PATH.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


add_src_to_syspath()


__all__ = ["add_src_to_syspath"]
