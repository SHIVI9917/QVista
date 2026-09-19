"""
Console widget.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTextEdit


class Console(QTextEdit):
    """Application console."""

    def __init__(self) -> None:
        super().__init__()

        self.setReadOnly(True)

        self.append("QVista Console")
        self.append("---------------------------")
        self.append("Application started.")
