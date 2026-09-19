"""
Application bootstrap for QVista.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme


def create_application() -> QApplication:
    """Create and configure the QApplication instance."""

    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION)

    # Every window in the program is dressed from here, including the
    # ones opened later: a stylesheet set on the application applies to
    # widgets created after it as well.
    theme.apply(app)

    return app
