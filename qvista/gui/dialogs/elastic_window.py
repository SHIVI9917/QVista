"""
Floating window for the elastic tensor.

A 6x6 matrix needs square-ish space to be readable, which the bottom dock
strip cannot give it, so it lives in its own window beside the main one.
"""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout

from qvista.gui.dialogs.side_window import SideWindow
from qvista.gui.widgets.elastic_panel import ElasticPanel


class ElasticWindow(SideWindow):
    """Hosts the elastic tensor panel."""

    def __init__(self, parent=None) -> None:

        super().__init__("QVista - Elastic Tensor", parent, preferred_size=760)

        self.panel = ElasticPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.panel)

    # ------------------------------------------------------------

    def show_result(self, result) -> None:
        """Populate the panel from an elastic result."""

        self.panel.show_result(result)

    def clear(self) -> None:
        """Empty the panel."""

        self.panel.clear()
