"""
K-points dialogs: pick a mesh by KSPACING, or set nx/ny/nz directly.

Both stay open after showing a result -- a preset or a spinbox change just
redraws the preview in place -- so a few options can be compared before
saving. Only "Close" dismisses either dialog.

``adopt=True`` is for callers that want the mesh rather than a file: the
save button becomes "Use this mesh" and accepts the dialog, and the
caller reads :attr:`grid`. The phonon workflow uses it to put the same
KPOINTS into every run folder at once, which is not something a save
dialog can express.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from qvista.analyses.kpoints import (
    CONVENTIONS,
    KP_RESOLUTION,
    RESOLUTION_PRESETS,
    TWO_PI,
    achieved_spacing,
    grid_for,
    kpoints_content,
)
from qvista.gui.dialogs.kpath_dialog import (
    PRIMARY_BUTTON_STYLE,
    SUCCESS_BUTTON_STYLE,
)
from qvista.gui.dialogs.plot_window import FigurePlaceholder, FigureWindow
from qvista.visualization.kpoints_plots import draw_kpoint_grid

#: Smallest and largest subdivision the Customised dialog allows per axis.
MIN_SUBDIVISIONS = 1
MAX_SUBDIVISIONS = 30


def _save_kpoints(parent, nx: int, ny: int, nz: int, scheme: str) -> None:
    """Shared "Save KPOINTS..." behaviour for both dialogs below."""

    filename, _ = QFileDialog.getSaveFileName(
        parent,
        "Save KPOINTS",
        "KPOINTS",
        "KPOINTS File (KPOINTS);;All Files (*)",
    )

    if not filename:
        return

    try:
        Path(filename).write_text(
            kpoints_content(nx, ny, nz, scheme), encoding="utf-8"
        )

    except OSError as exc:
        QMessageBox.critical(
            parent, "Save failed", f"Could not write {filename}: {exc}"
        )
        return

    QMessageBox.information(parent, "Saved", f"KPOINTS written to {filename}")


# ------------------------------------------------------------
# By KSPACING
# ------------------------------------------------------------


class KSpacingDialog(QDialog):
    """
    Pick a target spacing and see the resulting mesh.

    ``convention`` says how the number is meant -- a k-point resolution
    in units of ``2*pi/angstrom``, which is what papers quote, or VASP's
    KSPACING tag in inverse angstrom. They differ by a factor of
    ``2*pi`` along every axis, so reading one as the other is a factor of
    250 in the number of k-points; the convention is therefore stated on
    the row of buttons rather than left to the caller's memory.

    Nothing closes the dialog, so several presets can be tried before
    saving.
    """

    def __init__(
        self,
        structure,
        scheme: str,
        parent=None,
        adopt: bool = False,
        presets: Sequence[float] = RESOLUTION_PRESETS,
        convention: str = KP_RESOLUTION,
    ) -> None:

        super().__init__(parent)

        self.structure = structure
        self.scheme = scheme
        self.adopt = adopt
        self.convention = convention
        self._grid: tuple[int, int, int] | None = None

        self.setWindowTitle(f"K-points ({scheme}) by spacing")

        # No preview when the mesh is being adopted rather than saved:
        # the caller wants a number, and drawing one is what makes this
        # slow. A dense mesh on a supercell is tens of thousands of
        # points, and scattering them in 3D takes far longer than the
        # calculation it is a preview of.
        self.setMinimumSize(560, 200 if adopt else 620)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"Scheme: {scheme}"))

        row = QHBoxLayout()
        row.addWidget(
            QLabel(f"{CONVENTIONS.get(convention, 'Spacing')}:")
        )

        for value in presets:

            button = QPushButton(f"{value:g}")
            button.setStyleSheet(PRIMARY_BUTTON_STYLE)
            button.setToolTip(self._describe(value))
            button.clicked.connect(partial(self._compute, value))

            row.addWidget(button)

        row.addStretch(1)

        layout.addLayout(row)

        self.result_label = QLabel("Pick a spacing above.")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-weight: bold;")

        layout.addWidget(self.result_label)

        self.plot_panel = None
        self.figures = None

        if not adopt:

            # The mesh sits inside the Brillouin zone, which is a 3D
            # object drawn on a 2D axis and needs roughly square space
            # to be read; the dialog is a column of spin boxes. It gets
            # a window of its own.
            self.figures = FigureWindow(
                "QVista - K-point Mesh", ("K-point mesh",), self
            )

            self.plot_panel = self.figures.panel_for()

            layout.addWidget(FigurePlaceholder(self.figures, self))

        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.save_button = QPushButton(
            "Use this mesh" if adopt else "Save KPOINTS..."
        )
        self.save_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(
            self.accept if adopt else self._save
        )

        button_row.addWidget(self.save_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        button_row.addWidget(close_button)

        layout.addLayout(button_row)

    # ------------------------------------------------------------

    def _describe(self, value: float) -> str:
        """What one preset asks for, in inverse angstrom either way."""

        if self.convention == KP_RESOLUTION:

            return (
                f"{value:g} x 2π = {TWO_PI * value:.3f} Å⁻¹ between "
                f"neighbouring k-points."
            )

        return f"{value:g} Å⁻¹ between neighbouring k-points."

    def _compute(self, spacing: float) -> None:
        """Compute the mesh for one preset, and show it."""

        nx, ny, nz = grid_for(self.structure, spacing, self.convention)

        self._grid = (nx, ny, nz)

        # What was asked for beside what the mesh delivers. The two
        # differ because subdivisions are whole numbers, and on a long
        # axis one subdivision is most of the difference.
        got = achieved_spacing(self.structure, self._grid)

        self.result_label.setText(
            f"{CONVENTIONS.get(self.convention, 'Spacing')} = "
            f"{spacing:g}   →   {nx} x {ny} x {nz} = {nx * ny * nz} "
            f"k-points\n"
            f"Δk = {got[0]:.3f}, {got[1]:.3f}, {got[2]:.3f} Å⁻¹   "
            f"(asked for {self._target(spacing):.3f} Å⁻¹)"
        )

        if self.plot_panel is not None:

            draw_kpoint_grid(
                self.plot_panel.figure,
                nx,
                ny,
                nz,
                lattice=self.structure,
                scheme=self.scheme,
                title=f"{self.scheme} mesh: {nx} x {ny} x {nz}",
            )
            self.plot_panel.refresh()

            if self.figures is not None:
                self.figures.show_figure(reference=self)

        self.save_button.setEnabled(True)

    def _target(self, spacing: float) -> float:
        """The requested spacing in inverse angstrom, whatever was typed."""

        if self.convention == KP_RESOLUTION:
            return TWO_PI * spacing

        return spacing

    @property
    def grid(self) -> tuple[int, int, int] | None:
        """The mesh last computed, or ``None`` before any preset."""

        return self._grid

    def _save(self) -> None:
        """Write the last computed mesh to a KPOINTS file."""

        if self._grid is None:
            return

        nx, ny, nz = self._grid

        _save_kpoints(self, nx, ny, nz, self.scheme)


# ------------------------------------------------------------
# Customised
# ------------------------------------------------------------


class CustomisedKpointsDialog(QDialog):
    """Set nx, ny, nz directly and preview the mesh as they change."""

    def __init__(
        self,
        structure,
        scheme: str,
        parent=None,
        adopt: bool = False,
        start: tuple[int, int, int] | None = None,
    ) -> None:

        super().__init__(parent)

        self.structure = structure
        self.scheme = scheme
        self.adopt = adopt

        self.setWindowTitle(f"K-points ({scheme}) customised")

        # See KSpacingDialog: no preview when the mesh is being adopted.
        self.setMinimumSize(560, 220 if adopt else 620)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"Scheme: {scheme}"))

        form = QFormLayout()

        self.nx_box = QSpinBox()
        self.ny_box = QSpinBox()
        self.nz_box = QSpinBox()

        # A caller that already has a mesh in mind -- the phonon
        # workflow derives one for the supercell -- starts from it
        # rather than from 1 1 1, which nobody wants.
        opening = start or (1, 1, 1)

        for box, value in zip(
            (self.nx_box, self.ny_box, self.nz_box), opening
        ):
            box.setRange(MIN_SUBDIVISIONS, MAX_SUBDIVISIONS)
            box.setValue(
                max(MIN_SUBDIVISIONS, min(MAX_SUBDIVISIONS, int(value)))
            )

        form.addRow("nx", self.nx_box)
        form.addRow("ny", self.ny_box)
        form.addRow("nz", self.nz_box)

        layout.addLayout(form)

        self.result_label = QLabel()
        self.result_label.setStyleSheet("font-weight: bold;")

        layout.addWidget(self.result_label)

        self.plot_panel = None
        self.figures = None

        if not adopt:

            # The mesh sits inside the Brillouin zone, which is a 3D
            # object drawn on a 2D axis and needs roughly square space
            # to be read; the dialog is a column of spin boxes. It gets
            # a window of its own.
            self.figures = FigureWindow(
                "QVista - K-point Mesh", ("K-point mesh",), self
            )

            self.plot_panel = self.figures.panel_for()

            layout.addWidget(FigurePlaceholder(self.figures, self))

        layout.addStretch(1)

        for box in (self.nx_box, self.ny_box, self.nz_box):
            box.valueChanged.connect(self._refresh)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        save_button = QPushButton(
            "Use this mesh" if adopt else "Save KPOINTS..."
        )
        save_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        save_button.clicked.connect(
            self.accept if adopt else self._save
        )

        button_row.addWidget(save_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        button_row.addWidget(close_button)

        layout.addLayout(button_row)

        self._refresh()

    # ------------------------------------------------------------

    def _refresh(self) -> None:
        """Show, and where there is a preview redraw, the current mesh."""

        nx, ny, nz = self.grid

        got = achieved_spacing(self.structure, self.grid)

        self.result_label.setText(
            f"K-points: {nx} x {ny} x {nz} = {nx * ny * nz} in the grid\n"
            f"Δk = {got[0]:.3f}, {got[1]:.3f}, {got[2]:.3f} Å⁻¹"
        )

        if self.plot_panel is None:
            return

        draw_kpoint_grid(
            self.plot_panel.figure,
            nx,
            ny,
            nz,
            lattice=self.structure,
            scheme=self.scheme,
            title=f"{self.scheme} mesh: {nx} x {ny} x {nz}",
        )
        self.plot_panel.refresh()

        if self.figures is not None:
            self.figures.show_figure(reference=self)

    @property
    def grid(self) -> tuple[int, int, int]:
        """The mesh currently set."""

        return (
            self.nx_box.value(),
            self.ny_box.value(),
            self.nz_box.value(),
        )

    def _save(self) -> None:
        """Write the current nx/ny/nz to a KPOINTS file."""

        _save_kpoints(self, *self.grid, self.scheme)
