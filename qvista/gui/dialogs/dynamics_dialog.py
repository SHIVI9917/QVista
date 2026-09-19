"""
Setup dialogs for NEB and molecular dynamics.

Both runs are long, so both dialogs price the job before it starts: the
number of force evaluations is what the wait is made of, and it is far
cheaper to see it here than to discover it an hour in.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from qvista.analyses.guards import format_duration
from qvista.gui import theme
from qvista.analyses.md import (
    DEFAULT_FRICTION,
    DEFAULT_STEPS as MD_STEPS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMESTEP,
    ENSEMBLE_LABELS,
    ENSEMBLE_NVT,
)
from qvista.analyses.neb import (
    DEFAULT_FMAX,
    DEFAULT_IMAGES,
    DEFAULT_STEPS as NEB_STEPS,
)

INFO_STYLE = theme.INFO_STYLE
ERROR_STYLE = theme.ERROR_STYLE

#: Very rough cost of one force evaluation, per atom. Only used to give an
#: order of magnitude before a long run.
SECONDS_PER_ATOM = 0.02


def _estimate(evaluations: int, natoms: int) -> str:
    """Human-readable size of a pending run."""

    seconds = evaluations * natoms * SECONDS_PER_ATOM

    return (
        f"{evaluations:,} force evaluations x {natoms:,} atoms\n"
        f"Estimated run time: {format_duration(seconds * 0.5)} - "
        f"{format_duration(seconds * 3.0)}"
    )


# ------------------------------------------------------------
# NEB
# ------------------------------------------------------------


class NEBDialog(QDialog):
    """Choose the final structure and the path settings."""

    def __init__(self, natoms: int, start_name: str, parent=None) -> None:

        super().__init__(parent)

        self.setWindowTitle("NEB - minimum energy path")
        self.setMinimumWidth(560)

        self.natoms = natoms

        self.final_path: Path | None = None

        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(
                f"Start structure: {start_name}\n"
                f"Choose the end structure. It must contain the same atoms "
                f"in the same order."
            )
        )

        # ----------------------------------------------------------
        # End structure
        # ----------------------------------------------------------

        row = QHBoxLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("End structure file...")
        self.path_edit.setReadOnly(True)

        row.addWidget(self.path_edit)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        layout.addLayout(row)

        # ----------------------------------------------------------
        # Settings
        # ----------------------------------------------------------

        form = QFormLayout()

        self.images_box = QSpinBox()
        self.images_box.setRange(1, 30)
        self.images_box.setValue(DEFAULT_IMAGES)
        self.images_box.valueChanged.connect(self._refresh)

        form.addRow("Intermediate images", self.images_box)

        self.steps_box = QSpinBox()
        self.steps_box.setRange(1, 2000)
        self.steps_box.setValue(NEB_STEPS)
        self.steps_box.valueChanged.connect(self._refresh)

        form.addRow("Max optimiser steps", self.steps_box)

        self.fmax_box = QDoubleSpinBox()
        self.fmax_box.setDecimals(3)
        self.fmax_box.setRange(0.001, 1.0)
        self.fmax_box.setSingleStep(0.01)
        self.fmax_box.setValue(DEFAULT_FMAX)
        self.fmax_box.setSuffix(" eV/Å")

        form.addRow("Force convergence", self.fmax_box)

        layout.addLayout(form)

        self.climb_box = QCheckBox(
            "Climbing image (accurate saddle point)"
        )
        self.climb_box.setChecked(True)

        layout.addWidget(self.climb_box)

        # ----------------------------------------------------------
        # Cost
        # ----------------------------------------------------------

        self.cost = QLabel()
        self.cost.setTextFormat(Qt.PlainText)
        self.cost.setStyleSheet(INFO_STYLE)

        layout.addWidget(self.cost)

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

        self._refresh()

    # ------------------------------------------------------------

    def _browse(self) -> None:
        """Pick the end structure."""

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "End structure",
            "",
            (
                "Structure Files (POSCAR CONTCAR *.vasp *.cif *.xyz);;"
                "All Files (*)"
            ),
        )

        if not filename:
            return

        self.final_path = Path(filename)
        self.path_edit.setText(filename)

        self._refresh()

    def _refresh(self) -> None:
        """Update the cost estimate and the Run button."""

        # Every image is evaluated at every optimiser step.
        evaluations = self.images_box.value() * self.steps_box.value()

        self.cost.setStyleSheet(INFO_STYLE)
        self.cost.setText(_estimate(evaluations, self.natoms))

        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            self.final_path is not None
        )

    # ------------------------------------------------------------

    @property
    def n_images(self) -> int:
        return self.images_box.value()

    @property
    def steps(self) -> int:
        return self.steps_box.value()

    @property
    def fmax(self) -> float:
        return self.fmax_box.value()

    @property
    def climb(self) -> bool:
        return self.climb_box.isChecked()


# ------------------------------------------------------------
# Molecular dynamics
# ------------------------------------------------------------


class MDDialog(QDialog):
    """Choose the ensemble, temperature and length of a run."""

    def __init__(self, natoms: int, parent=None) -> None:

        super().__init__(parent)

        self.setWindowTitle("Molecular dynamics")
        self.setMinimumWidth(520)

        self.natoms = natoms

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.ensemble_box = QComboBox()

        for key, label in ENSEMBLE_LABELS.items():
            self.ensemble_box.addItem(label, key)

        self.ensemble_box.setCurrentIndex(
            self.ensemble_box.findData(ENSEMBLE_NVT)
        )
        self.ensemble_box.currentIndexChanged.connect(self._refresh)

        form.addRow("Ensemble", self.ensemble_box)

        self.temperature_box = QDoubleSpinBox()
        self.temperature_box.setRange(0.0, 5000.0)
        self.temperature_box.setSingleStep(50.0)
        self.temperature_box.setValue(DEFAULT_TEMPERATURE)
        self.temperature_box.setSuffix(" K")

        form.addRow("Temperature", self.temperature_box)

        self.timestep_box = QDoubleSpinBox()
        self.timestep_box.setDecimals(2)
        self.timestep_box.setRange(0.1, 10.0)
        self.timestep_box.setSingleStep(0.25)
        self.timestep_box.setValue(DEFAULT_TIMESTEP)
        self.timestep_box.setSuffix(" fs")
        self.timestep_box.valueChanged.connect(self._refresh)

        form.addRow("Timestep", self.timestep_box)

        self.steps_box = QSpinBox()
        self.steps_box.setRange(1, 100000)
        self.steps_box.setValue(MD_STEPS)
        self.steps_box.valueChanged.connect(self._refresh)

        form.addRow("Steps", self.steps_box)

        self.friction_box = QDoubleSpinBox()
        self.friction_box.setDecimals(4)
        self.friction_box.setRange(0.0001, 1.0)
        self.friction_box.setSingleStep(0.005)
        self.friction_box.setValue(DEFAULT_FRICTION)
        self.friction_box.setSuffix(" /fs")

        form.addRow("Langevin friction", self.friction_box)

        layout.addLayout(form)

        # ----------------------------------------------------------
        # Cost and warnings
        # ----------------------------------------------------------

        self.cost = QLabel()
        self.cost.setTextFormat(Qt.PlainText)
        self.cost.setStyleSheet(INFO_STYLE)
        self.cost.setWordWrap(True)

        layout.addWidget(self.cost)

        self._build_outputs(layout)

        # ----------------------------------------------------------

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        buttons.button(QDialogButtonBox.Ok).setText("Run")

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self._refresh()

    # ------------------------------------------------------------
    # Where the run goes
    # ------------------------------------------------------------

    def _build_outputs(self, layout: QVBoxLayout) -> None:
        """
        Somewhere for the trajectory to live after the window closes.

        A run used to exist only in the session: everything under
        MD > CHGNet > Analysis opened on it from memory, so closing the
        window or restarting meant running the dynamics again to ask a
        second question of them. Written out, the folder reopens through
        MD > VASP > Analysis and the trajectory outlives the session.

        Checked by default, because the run is the expensive part and
        the files are not.
        """

        from qvista.services import md_vasp

        box = QGroupBox("Save the run")

        column = QVBoxLayout(box)
        column.setSpacing(6)

        where = QHBoxLayout()
        where.setSpacing(6)

        self.save_outputs = QCheckBox("Save to")
        self.save_outputs.setChecked(True)

        where.addWidget(self.save_outputs)

        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText(
            "beside the structure if left empty"
        )

        where.addWidget(self.output_folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_output)

        where.addWidget(browse)

        column.addLayout(where)

        what = QHBoxLayout()
        what.setSpacing(12)

        #: Keyed by the argument name ``save_run`` takes for each.
        self.save_what: dict[str, QCheckBox] = {}

        for key, label in md_vasp.SAVEABLE:

            item = QCheckBox(label)
            item.setChecked(True)

            self.save_what[key] = item

            what.addWidget(item)

        what.addStretch(1)

        column.addLayout(what)

        self.save_outputs.toggled.connect(self._outputs_toggled)

        layout.addWidget(box)

    def _outputs_toggled(self, on: bool) -> None:

        self.output_folder.setEnabled(on)

        for item in self.save_what.values():
            item.setEnabled(on)

    def _browse_output(self) -> None:

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Save the dynamics run in",
            self.output_folder.text(),
        )

        if chosen:
            self.output_folder.setText(chosen)

    @property
    def save_requested(self) -> bool:
        return self.save_outputs.isChecked()

    @property
    def output_directory(self) -> str:
        return self.output_folder.text().strip()

    @property
    def save_choices(self) -> dict[str, bool]:
        return {
            key: item.isChecked()
            for key, item in self.save_what.items()
        }

    # ------------------------------------------------------------

    def _refresh(self) -> None:
        """Update the estimate and warn about an unsafe timestep."""

        steps = self.steps_box.value()

        text = _estimate(steps, self.natoms)

        text += (
            f"\nSimulated time: "
            f"{steps * self.timestep_box.value():,.0f} fs"
        )

        # 2 fs is about the limit for anything containing hydrogen; beyond
        # 3 fs the integration is unreliable for most solids.
        if self.timestep_box.value() > 2.0:
            self.cost.setStyleSheet(ERROR_STYLE)
            text += (
                "\n\nA timestep above 2 fs is unsafe for most systems and "
                "will not conserve energy. Check the drift reported at the "
                "end, or reduce it."
            )
        else:
            self.cost.setStyleSheet(INFO_STYLE)

        self.cost.setText(text)

    # ------------------------------------------------------------

    @property
    def ensemble(self) -> str:
        return self.ensemble_box.currentData()

    @property
    def temperature(self) -> float:
        return self.temperature_box.value()

    @property
    def timestep(self) -> float:
        return self.timestep_box.value()

    @property
    def steps(self) -> int:
        return self.steps_box.value()

    @property
    def friction(self) -> float:
        return self.friction_box.value()
