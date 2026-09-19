"""
Supercell selection for the phonon workflow.

A phonon run's cost is set almost entirely by the supercell, and a blind
3x3x3 is dangerous on real user files: on a slab it replicates vacuum, and
at P1 there is no symmetry to reduce the displacement count. This dialog
shows what the choice actually costs -- using phonopy's real displacement
count, not a guess -- and refuses to enable the run until that has been
looked at.
"""

from __future__ import annotations

from typing import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from qvista.gui import theme
from qvista.analyses.guards import estimate_cost
from qvista.analyses.phonon import SUPERCELL_ATOM_LIMIT

ERROR_STYLE = theme.ERROR_STYLE
WARNING_STYLE = theme.WARNING_STYLE
INFO_STYLE = theme.INFO_STYLE


class SupercellDialog(QDialog):
    """
    Choose the supercell and confirm the cost of the run.

    ``count_displacements`` is injected: it takes a supercell and returns
    ``(displacement_count, atoms_per_supercell)`` by actually calling
    phonopy, so the number shown is the number that will be run.
    """

    def __init__(
        self,
        formula: str,
        primitive_natoms: int,
        lattice_abc: Sequence[float],
        lattice_angles: Sequence[float],
        primitive_volume: float,
        count_displacements: Callable[[Sequence[int]], tuple[int, int]],
        suggested: Sequence[int] = (2, 2, 2),
        warnings: Sequence[str] = (),
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.setWindowTitle("Phonon supercell")
        self.setMinimumWidth(560)

        self.formula = formula
        self.primitive_natoms = primitive_natoms
        self.lattice_abc = list(lattice_abc)
        self.lattice_angles = list(lattice_angles)
        self.primitive_volume = primitive_volume

        self._count_displacements = count_displacements

        #: Set once the user has actually looked at the cost.
        self.estimated_for: tuple[int, int, int] | None = None

        self.n_displacements = 0
        self.supercell_natoms = 0

        layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        # Guard warnings
        # ----------------------------------------------------------

        if warnings:

            for text in warnings:

                label = QLabel(text)
                label.setWordWrap(True)
                label.setStyleSheet(WARNING_STYLE)

                layout.addWidget(label)

        # ----------------------------------------------------------
        # Supercell
        # ----------------------------------------------------------

        box = QGroupBox("Supercell")

        row = QHBoxLayout(box)

        self.spins: list[QSpinBox] = []

        for axis, value in zip("abc", suggested):

            row.addWidget(QLabel(f"N{axis}"))

            spin = QSpinBox()
            spin.setRange(1, 10)
            spin.setValue(int(value))
            spin.valueChanged.connect(self._on_supercell_changed)

            row.addWidget(spin)

            self.spins.append(spin)

        row.addStretch(1)

        layout.addWidget(box)

        # ----------------------------------------------------------
        # Cell summary
        # ----------------------------------------------------------

        self.summary = QLabel()
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setStyleSheet(INFO_STYLE)

        layout.addWidget(self.summary)

        # ----------------------------------------------------------
        # Cost
        # ----------------------------------------------------------

        cost_row = QHBoxLayout()

        self.estimate_button = QPushButton("Calculate cost")
        self.estimate_button.clicked.connect(self._estimate)

        cost_row.addWidget(self.estimate_button)
        cost_row.addStretch(1)

        layout.addLayout(cost_row)

        self.cost = QLabel()
        self.cost.setTextFormat(Qt.PlainText)
        self.cost.setWordWrap(True)

        layout.addWidget(self.cost)

        self.override = QCheckBox(
            "Run anyway - I understand this is very large"
        )
        self.override.setVisible(False)
        self.override.stateChanged.connect(self._update_run_enabled)

        layout.addWidget(self.override)

        # ----------------------------------------------------------
        # Buttons
        # ----------------------------------------------------------

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        self.buttons.button(QDialogButtonBox.Ok).setText("Run")

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(self.buttons)

        # ----------------------------------------------------------

        self._on_supercell_changed()

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    @property
    def supercell(self) -> tuple[int, int, int]:
        """The chosen supercell."""

        return tuple(spin.value() for spin in self.spins)

    def _multiplicity(self) -> int:
        """How many primitive cells the supercell contains."""

        nx, ny, nz = self.supercell

        return nx * ny * nz

    # ------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------

    def _on_supercell_changed(self) -> None:
        """Refresh the summary; any previous estimate no longer applies."""

        multiplicity = self._multiplicity()

        a, b, c = self.lattice_abc
        alpha, beta, gamma = self.lattice_angles

        nx, ny, nz = self.supercell

        self.summary.setText(
            "\n".join(
                [
                    f"Formula (primitive)  : {self.formula}",
                    f"Primitive atoms      : {self.primitive_natoms}",
                    f"Multiplicity         : {nx} x {ny} x {nz}"
                    f" = {multiplicity}",
                    f"Supercell atoms      : "
                    f"{self.primitive_natoms * multiplicity}",
                    f"Primitive lattice    : "
                    f"a={a:.4f}  b={b:.4f}  c={c:.4f} Å",
                    f"Primitive angles     : "
                    f"{alpha:.2f}  {beta:.2f}  {gamma:.2f} deg",
                    f"Supercell lattice    : "
                    f"a={a * nx:.4f}  b={b * ny:.4f}  c={c * nz:.4f} Å",
                    f"Primitive volume     : {self.primitive_volume:.4f} Å³",
                    f"Supercell volume     : "
                    f"{self.primitive_volume * multiplicity:.4f} Å³",
                ]
            )
        )

        # A stale estimate must not authorise a different run.
        if self.estimated_for != self.supercell:

            self.cost.setStyleSheet(INFO_STYLE)
            self.cost.setText(
                "Press Calculate cost to see how many force evaluations "
                "this supercell needs."
            )

            self.override.setVisible(False)
            self.override.setChecked(False)

        self._update_run_enabled()

    def _estimate(self) -> None:
        """Ask phonopy how many displacements this supercell really needs."""

        supercell = self.supercell

        self.estimate_button.setEnabled(False)
        self.cost.setStyleSheet(INFO_STYLE)
        self.cost.setText("Generating displacements...")

        try:
            n_displacements, natoms = self._count_displacements(supercell)

        except Exception as exc:
            self.estimated_for = None
            self.cost.setStyleSheet(ERROR_STYLE)
            self.cost.setText(f"Could not determine the cost: {exc}")
            self.estimate_button.setEnabled(True)
            self._update_run_enabled()
            return

        self.estimate_button.setEnabled(True)

        self.estimated_for = supercell
        self.n_displacements = n_displacements
        self.supercell_natoms = natoms

        text = estimate_cost(n_displacements, natoms)

        too_big = natoms > SUPERCELL_ATOM_LIMIT

        self.override.setVisible(too_big)

        if too_big:

            self.cost.setStyleSheet(ERROR_STYLE)

            self.cost.setText(
                f"{text}\n\n"
                f"{natoms} atoms per displaced supercell is above the "
                f"{SUPERCELL_ATOM_LIMIT}-atom limit for an interactive run. "
                f"Reduce the supercell -- especially along any vacuum "
                f"direction -- or tick the box below to proceed anyway."
            )

        else:
            self.cost.setStyleSheet(INFO_STYLE)
            self.cost.setText(text)

        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        """Run stays disabled until the cost has been seen and accepted."""

        ok = self.buttons.button(QDialogButtonBox.Ok)

        if self.estimated_for != self.supercell:
            ok.setEnabled(False)
            return

        if self.supercell_natoms > SUPERCELL_ATOM_LIMIT:
            ok.setEnabled(self.override.isChecked())
            return

        ok.setEnabled(True)
