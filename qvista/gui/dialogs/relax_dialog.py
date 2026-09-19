"""
Setup dialog for a structure relaxation.

Short, because there are only three decisions: what may move, how small
the forces have to get, and when to give up. The dialog prices the run
the same way the NEB and MD ones do -- an optimiser step is one force
evaluation, and that is what the wait is made of.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from qvista.analyses.guards import format_duration
from qvista.analyses.relax import (
    CONSTANT_VOLUME,
    DEFAULT_FMAX,
    DEFAULT_STEPS,
    IONS_AND_CELL,
    IONS_ONLY,
    ISIF,
    MODES,
    NEEDS_CELL,
)
from qvista.gui import theme

#: Very rough cost of one force evaluation, per atom. Only used to give
#: an order of magnitude before a long run.
SECONDS_PER_ATOM = 0.02


class RelaxDialog(QDialog):
    """Choose what may move, and how far to push it."""

    def __init__(
        self, natoms: int, periodic: bool = True, parent=None
    ) -> None:

        super().__init__(parent)

        self.setWindowTitle("Relax structure")
        self.setMinimumWidth(520)

        self.natoms = natoms
        self.periodic = periodic

        layout = QVBoxLayout(self)

        form = QFormLayout()

        # ----------------------------------------------------------
        # What may move
        # ----------------------------------------------------------

        self.mode_box = QComboBox()

        for key, label in MODES.items():
            self.mode_box.addItem(label, key)

        # A molecule has no cell to relax, so the choice is not offered.
        self.mode_box.setCurrentIndex(
            self.mode_box.findData(
                IONS_AND_CELL if periodic else IONS_ONLY
            )
        )

        if not periodic:

            # A molecule has no cell, so only ISIF 2 applies. The two
            # cell modes are greyed rather than the whole control: the
            # box still has to say which one is in force.
            for index in range(self.mode_box.count()):

                if self.mode_box.itemData(index) in NEEDS_CELL:

                    item = self.mode_box.model().item(index)

                    if item is not None:
                        item.setEnabled(False)

        self.mode_box.currentIndexChanged.connect(self._refresh)

        form.addRow("Relax", self.mode_box)

        # ----------------------------------------------------------
        # How far
        # ----------------------------------------------------------

        self.fmax_box = QDoubleSpinBox()
        self.fmax_box.setDecimals(3)
        self.fmax_box.setRange(0.001, 1.0)
        self.fmax_box.setSingleStep(0.005)
        self.fmax_box.setValue(DEFAULT_FMAX)
        self.fmax_box.setSuffix(" eV/A")
        self.fmax_box.setToolTip(
            "The relaxation stops when no atom feels a force larger "
            "than this."
        )
        self.fmax_box.valueChanged.connect(self._refresh)

        form.addRow("Force target", self.fmax_box)

        self.steps_box = QSpinBox()
        self.steps_box.setRange(1, 10000)
        self.steps_box.setValue(DEFAULT_STEPS)
        self.steps_box.setToolTip(
            "A cap, not a target. A relaxation that has not converged "
            "in this many steps is unlikely to."
        )
        self.steps_box.valueChanged.connect(self._refresh)

        form.addRow("Maximum steps", self.steps_box)

        layout.addLayout(form)

        # ----------------------------------------------------------

        self.cost = QLabel("")
        self.cost.setWordWrap(True)

        layout.addWidget(self.cost)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self._refresh()

    # ------------------------------------------------------------

    def _refresh(self) -> None:
        """Price the run, and say what the choices mean."""

        steps = self.steps_box.value()

        seconds = steps * self.natoms * SECONDS_PER_ATOM

        text = (
            f"Up to {steps:,} force evaluations x {self.natoms:,} atoms\n"
            f"Estimated: {format_duration(seconds * 0.5)} - "
            f"{format_duration(seconds * 3.0)}, and usually far less -- "
            f"the cap is only reached when it fails to converge."
        )

        if self.mode == IONS_AND_CELL:
            text += (
                "\n\nISIF 3: the cell moves too, shape and volume. "
                "Expect the volume to shift by a per cent or two -- "
                "that is the force field disagreeing with whatever "
                "produced the input geometry."
            )

        elif self.mode == CONSTANT_VOLUME:
            text += (
                "\n\nISIF 4: the cell may change shape but not size. "
                "For relaxing at a fixed density -- along an equation "
                "of state, or at a volume chosen elsewhere. The volume "
                "reported afterwards should be unchanged, which makes "
                "it a check rather than a result."
            )

        else:
            text += (
                "\n\nISIF 2: the cell is held exactly as given. For "
                "lattice constants taken from experiment, or from a "
                "calculation trusted more than this one."
            )

        if self.fmax_box.value() < 0.005:
            self.cost.setStyleSheet(theme.ERROR_STYLE)
            text += (
                "\n\nA target this tight is below the force field's own "
                "accuracy. It will spend a long time chasing noise."
            )
        else:
            self.cost.setStyleSheet(theme.INFO_STYLE)

        self.cost.setText(text)

    # ------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.mode_box.currentData()

    @property
    def fmax(self) -> float:
        return self.fmax_box.value()

    @property
    def steps(self) -> int:
        return self.steps_box.value()
