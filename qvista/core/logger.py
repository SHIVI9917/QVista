"""
Logging configuration for QVista.
"""

from __future__ import annotations

import logging


def setup_logger() -> None:
    """Configure the root logger."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
