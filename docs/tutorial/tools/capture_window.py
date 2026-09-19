"""
Screenshot the main window, in the real interface.

The important detail is that the application is built through
:func:`create_application`, exactly as ``python -m qvista`` does, so the
theme stylesheet is applied and the screenshots show the program people
actually see. Building a bare ``QApplication`` instead -- which is what
produced the previous set of images -- gives unstyled Qt widgets and, on
several dialogs, white text on a white background.

The 3D viewport needs one extra step. ``QWidget.grab()`` copies Qt's
backing store, and VTK draws through a native OpenGL surface that never
enters it, so the viewport comes back solid black. The viewport is
therefore rendered separately through VTK's own screenshot and painted
into the grab. Compositing rather than grabbing the screen keeps this
deterministic: nothing depends on the window being frontmost, and
nothing else on the desktop can end up in the picture.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "images"
DEMO = Path(sys.argv[1] if len(sys.argv) > 1
            else "LiMnO2-1819592.vasp").resolve()

from qvista.app import create_application  # noqa: E402

app = create_application()

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402

from qvista.gui.main_window import MainWindow  # noqa: E402

WIDTH, HEIGHT = 1600, 1000


def pump(ms: int = 700) -> None:
    """Let Qt and VTK settle."""

    end = time.time() + ms / 1000

    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


window = MainWindow()
window.resize(WIDTH, HEIGHT)
window.show()
pump(1500)

viewer = window.workspace.viewer
panel = window.visualization

done: list[tuple[str, int]] = []


def capture(name: str, settle: int = 900) -> None:
    """Grab the window with the viewport painted in."""

    pump(settle)

    pixmap = window.grab()

    frame = viewer.screenshot(
        return_img=True, transparent_background=False
    )
    frame = np.ascontiguousarray(frame[:, :, :3])

    height, width, _ = frame.shape

    image = QImage(
        frame.data, width, height, 3 * width, QImage.Format_RGB888
    ).copy()

    corner = viewer.mapTo(window, QPoint(0, 0))

    # The pixmap carries a devicePixelRatio, so the painter is already
    # working in logical coordinates -- scaling the rectangle by the
    # ratio here would apply it twice.
    painter = QPainter(pixmap)
    painter.drawImage(
        QRect(corner.x(), corner.y(), viewer.width(), viewer.height()),
        image,
    )
    painter.end()

    path = OUT / f"{name}.png"
    pixmap.save(str(path))
    done.append((name, path.stat().st_size))
    print(f"  {name:26s} {path.stat().st_size // 1024:5d} KB", flush=True)


print("capturing:", flush=True)

# ------------------------------------------------------------------
# Before anything is loaded
# ------------------------------------------------------------------

capture("01-empty-window")

# ------------------------------------------------------------------
# A structure open, in the default representation
# ------------------------------------------------------------------

project = window.project_controller.open_project(str(DEMO))
window.update_project_ui(project)
pump(2000)

capture("02-structure-open")

# ------------------------------------------------------------------
# The three representations
# ------------------------------------------------------------------

panel.space_fill.setChecked(True)
capture("03-space-fill")

panel.polyhedral.setChecked(True)
capture("04-polyhedral")

panel.ball_stick.setChecked(True)
capture("05-ball-and-stick")

# ------------------------------------------------------------------
# Standard views
# ------------------------------------------------------------------

panel.x_button.click()
capture("06-view-x")

panel.z_button.click()
capture("07-view-z")

panel.iso_button.click()
capture("08-view-iso")

# ------------------------------------------------------------------
# Periodic images: the page says +X and +Y set to 2.00
# ------------------------------------------------------------------

panel.z_button.click()
pump(400)

panel.x_plus.setValue(2.00)
panel.y_plus.setValue(2.00)
window.update_boundary()

capture("09-boundary-expanded", settle=1600)

print(f"\ncaptured {len(done)} window shots", flush=True)
