"""
Application entry point.
"""

from __future__ import annotations

import os
import sys

# Work around VTK/PyVistaQt issues with Wayland on WSLg.
# Linux only -- "xcb" does not exist on Windows/macOS and forcing it
# stops Qt from creating any window at all.
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from qvista.app import create_application
from qvista.core.logger import setup_logger
from qvista.gui.main_window import MainWindow


def main() -> int:
    """Launch QVista."""

    setup_logger()

    app = create_application()

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
