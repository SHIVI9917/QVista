"""
Phonopy's VASP route, in three clicks.

Dimensions, then a method, then the inputs -- and nothing else on
screen. Every number that can be worked out is worked out: the cutoff
from the POTCARs' ENMAX, the k-mesh from the supercell. Anything that
goes wrong is a message box, not a line of red text somewhere on a
panel.

    Dimensions          nx, ny, nz.  Proceed generates the supercell
                        and the displaced cells.

    DFPT                writes DFPT/POSCAR from SPOSCAR.
    Finite Displacement writes Finite Displacement/Config_1/POSCAR, ...

    K-points, INCAR     optional: the same dialogs the K-points and
                        INCAR menu entries open, applied to every
                        folder at once instead of saving one file.

    Generate VASP       adds INCAR, KPOINTS and POTCAR to whichever
    Inputs              folders were just written.

Both files have a sensible default -- a mesh from VASP's KSPACING rule
on the supercell, and the INCAR for the chosen route -- so the two
buttons in the middle are there to override, not to fill in a blank.

:class:`VaspPostprocessDialog` is the other end: point it at that folder
once VASP has run.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qvista.analyses.kpoints import GAMMA_SCHEME, MP_SCHEME
from qvista.analyses.phonon import SUPERCELL_ATOM_LIMIT
from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.incar_dialog import CardButton
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.analyses.kpoints import VASP_KSPACING
from qvista.gui.dialogs.kpoints_dialog import (
    CustomisedKpointsDialog,
    KSpacingDialog,
)
from qvista.gui.dialogs.potcar_dialog import library_choices
from qvista.services import phonopy_vasp
from qvista.services.incar import ENCUT_FALLBACK, encut_for
from qvista.services.phonopy_vasp import (
    DEFAULT_KSPACING,
    KSPACING_PRESETS,
    METHODS,
)
from qvista.services.potcar import concatenate, ordered_elements

#: Where the folder last written to, and last read from, are remembered.
WRITE_KEY = "phonon/vasp_write_root"
READ_KEY = "phonon/vasp_read_root"

#: Default supercell, before the caller suggests otherwise.
DEFAULT_SUPERCELL = (2, 2, 2)

#: Sampling for the density of states, per direction.
DEFAULT_MESH_N = 20

#: q-points per band segment.
DEFAULT_BAND_POINTS = 51

#: The one status line each dialog carries.
STATUS_STYLE = f"color: {theme.TEXT_DIM};"


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:
    """Set the single line of feedback a dialog is allowed."""

    label.setStyleSheet(style)
    label.setText(text)


# ----------------------------------------------------------------
# Step one
# ----------------------------------------------------------------


class PhononIncarDialog(QDialog):
    """
    The phonon INCAR, shown as it will be written.

    Editable, because a cutoff or a smearing width is sometimes worth
    changing by hand -- but it opens as the file itself, not as a
    builder. Whatever is in the box is what goes into every run folder,
    line for line.
    """

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(parent)

        self.original = text

        self.setWindowTitle("INCAR")
        fit_to_screen(self, 560, 620)

        layout = QVBoxLayout(self)

        self.editor = QPlainTextEdit()
        self.editor.setPlainText(text)
        self.editor.setStyleSheet(theme.MONO_STYLE)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)

        layout.addWidget(self.editor, 1)

        buttons = QHBoxLayout()

        reset = QPushButton("Reset")
        reset.clicked.connect(
            lambda: self.editor.setPlainText(self.original)
        )

        buttons.addWidget(reset)
        buttons.addStretch(1)

        use = QPushButton("Use this INCAR")
        use.setStyleSheet(SUCCESS_BUTTON_STYLE)
        use.clicked.connect(self.accept)

        buttons.addWidget(use)

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        buttons.addWidget(cancel)

        layout.addLayout(buttons)

    def text(self) -> str:
        """What will be written."""

        return self.editor.toPlainText().rstrip() + "\n"


class DimensionsDialog(QDialog):
    """
    nx, ny, nz -- and Proceed or Cancel.

    The only thing asked before phonopy runs, because it is the only
    thing phonopy cannot work out: how far the force constants should be
    allowed to reach.
    """

    def __init__(self, suggested=DEFAULT_SUPERCELL, parent=None) -> None:

        super().__init__(parent)

        self.setWindowTitle("Dimensions")

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.spins: list[QSpinBox] = []

        for axis, value in zip(("nx", "ny", "nz"), suggested):

            row.addWidget(QLabel(axis))

            spin = QSpinBox()
            spin.setRange(1, 12)
            spin.setValue(int(value))
            spin.setMinimumWidth(64)
            spin.setAlignment(Qt.AlignCenter)

            row.addWidget(spin)

            self.spins.append(spin)

        row.addStretch(1)

        layout.addLayout(row)

        buttons = QDialogButtonBox()

        proceed = buttons.addButton(
            "Proceed", QDialogButtonBox.AcceptRole
        )
        proceed.setStyleSheet(SUCCESS_BUTTON_STYLE)

        buttons.addButton("Cancel", QDialogButtonBox.RejectRole)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    @property
    def supercell(self) -> tuple[int, int, int]:
        """The chosen repetition."""

        return tuple(spin.value() for spin in self.spins)


# ----------------------------------------------------------------
# Steps two and three
# ----------------------------------------------------------------


class PhononVaspDialog(QDialog):
    """
    Pick a route, then generate the inputs for it.

    ``phonon`` is a built Phonopy object with its displacements already
    generated -- everything on this screen writes files out of it and
    computes nothing.
    """

    def __init__(self, phonon, formula: str = "", parent=None) -> None:

        super().__init__(parent)

        self.phonon = phonon
        self.formula = formula or "structure"

        self.settings = QSettings(ORGANIZATION, APP_NAME)

        #: Chosen where the folders go, once, and reused if the method
        #: is changed afterwards.
        self.root: Path | None = None

        #: What the last method click wrote.
        self.config = None

        #: The method that wrote it.
        self.method = None

        #: Set by the k-points chooser, or left None to be derived.
        self.mesh: tuple[int, int, int] | None = None
        self.scheme: str = GAMMA_SCHEME

        #: Set by the INCAR dialog, or left empty to be derived.
        self.incar: str = ""
        self.incar_label: str = ""

        self.cards: dict[str, CardButton] = {}

        self.setWindowTitle("Phonopy - VASP Pre-Processing")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ----------------------------------------------------------

        self.summary = QLabel()
        self.summary.setStyleSheet(
            f"color: {theme.TEXT}; font-weight: 600;"
        )

        layout.addWidget(self.summary)

        layout.addWidget(_hline())

        # ----------------------------------------------------------

        cards = QHBoxLayout()
        cards.setSpacing(8)

        for method in METHODS:

            card = CardButton(method.name, method.tagline)
            card.picked.connect(self._pick)

            cards.addWidget(card, 1)

            self.cards[method.name] = card

        layout.addLayout(cards)

        # ----------------------------------------------------------

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)

        layout.addWidget(self.status)

        layout.addWidget(_hline())

        layout.addLayout(self._build_choosers())

        self.choices = QLabel()
        self.choices.setTextFormat(Qt.PlainText)
        self.choices.setStyleSheet(theme.MONO_STYLE)

        layout.addWidget(self.choices)

        layout.addStretch(1)

        # ----------------------------------------------------------

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.generate_button = QPushButton("Generate VASP Inputs")
        self.generate_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self._generate)

        buttons.addWidget(self.generate_button)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        buttons.addWidget(close)

        layout.addLayout(buttons)

        # ----------------------------------------------------------

        self._describe()

    # ------------------------------------------------------------
    # How the two files get made
    # ------------------------------------------------------------

    def _build_choosers(self) -> QHBoxLayout:
        """The K-points and INCAR buttons, between method and generate."""

        row = QHBoxLayout()
        row.setSpacing(8)

        self.kpoints_button = QPushButton("K-points")
        self.kpoints_button.setEnabled(False)

        menu = QMenu(self.kpoints_button)

        for scheme in (GAMMA_SCHEME, MP_SCHEME):

            submenu = menu.addMenu(scheme)

            for label, customised in (
                ("K-spacing", False),
                ("Customised", True),
            ):
                action = submenu.addAction(label)
                action.triggered.connect(
                    lambda _checked=False, s=scheme, c=customised: (
                        self._choose_kpoints(s, c)
                    )
                )

        self.kpoints_button.setMenu(menu)

        row.addWidget(self.kpoints_button)

        self.incar_button = QPushButton("INCAR")
        self.incar_button.setEnabled(False)
        self.incar_button.clicked.connect(self._choose_incar)

        row.addWidget(self.incar_button)

        row.addStretch(1)

        return row

    def _supercell_structure(self):
        """
        The supercell as pymatgen, which is what the mesh is for.

        Both k-points dialogs derive a grid from the reciprocal lattice
        of whatever they are given, and the runs are on the supercell --
        handing them the unit cell would oversample every run by the
        repetition factor.
        """

        return phonopy_vasp.to_pymatgen(self.phonon.supercell)

    def _derived_mesh(self) -> tuple[int, int, int]:
        """The mesh that will be used if nobody chooses one."""

        return phonopy_vasp.mesh_for(
            self._supercell_structure(), DEFAULT_KSPACING
        )

    def _choose_kpoints(self, scheme: str, customised: bool) -> None:
        """Open the same k-points dialog the K-points menu opens."""

        structure = self._supercell_structure()

        if customised:
            dialog = CustomisedKpointsDialog(
                structure,
                scheme,
                parent=self,
                adopt=True,
                start=self.mesh or self._derived_mesh(),
            )

        else:
            # These are VASP's KSPACING tag, in inverse angstrom with
            # the 2*pi in it -- not the K-points menu's resolution
            # presets, which are in units of 2*pi/angstrom and mean a
            # mesh 2*pi times denser along every axis. Both go through
            # the same dialog, so which one is meant has to be said.
            dialog = KSpacingDialog(
                structure,
                scheme,
                parent=self,
                adopt=True,
                presets=KSPACING_PRESETS,
                convention=VASP_KSPACING,
            )

        if dialog.exec() != QDialog.Accepted:
            return

        if dialog.grid is None:
            return

        self.mesh = tuple(dialog.grid)
        self.scheme = scheme

        self._show_choices()

    def _choose_incar(self) -> None:
        """Show the phonon INCAR, and let it be edited before it goes in."""

        if self.method is None:
            return

        dialog = PhononIncarDialog(
            self.incar or phonopy_vasp.phonon_incar(
                self.method, encut=self._encut()[0]
            ),
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            return

        self.incar = dialog.text()
        self.incar_label = "edited"

        self._show_choices()

    def _show_choices(self) -> None:
        """The two lines saying what will be written."""

        if self.method is None:
            self.choices.setText("")
            return

        mesh = self.mesh or self._derived_mesh()

        nx, ny, nz = mesh

        source = "chosen" if self.mesh else "derived"

        self.choices.setText(
            f"KPOINTS   {self.scheme} {nx} x {ny} x {nz}  ({source})\n"
            f"INCAR     Phonopy - {self.method.name}"
            + (f"  ({self.incar_label})" if self.incar_label else "")
        )

    # ------------------------------------------------------------

    def _describe(self) -> None:
        """The one line at the top: what phonopy produced."""

        nx, ny, nz = self.supercell

        count = len(self.phonon.supercells_with_displacements)

        self.summary.setText(
            f"{nx} x {ny} x {nz}  -  "
            f"{len(self.phonon.supercell)} atoms  -  "
            f"SPOSCAR + {count} displaced cells"
        )

        _status(self.status, "Choose DFPT or Finite Displacement.")

    @property
    def supercell(self) -> tuple[int, int, int]:
        """The repetition phonopy was built with."""

        return tuple(
            int(abs(self.phonon.supercell_matrix[axis][axis]))
            for axis in range(3)
        )

    # ------------------------------------------------------------
    # Step two: write the POSCARs
    # ------------------------------------------------------------

    def _pick(self, name: str) -> None:
        """A method card was clicked: write its folders."""

        method = phonopy_vasp.method_by_name(name)

        if method is None:
            return

        if self.root is None and not self._ask_where():
            return

        for title, card in self.cards.items():
            card.set_checked(title == name)

        self.method = method

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            self.config = phonopy_vasp.write_structures(
                self.root, self.phonon, method
            )

        except Exception as exc:
            QApplication.restoreOverrideCursor()

            self._warn(
                "Could not write the folders",
                f"{type(exc).__name__}: {exc}",
            )
            return

        finally:
            if QGuiApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        runs = self.config.force_runs

        where = (
            f"{phonopy_vasp.DFPT_DIR}/"
            if method.single_run
            else f"{phonopy_vasp.FD_DIR}/"
            f"{phonopy_vasp.config_name(1)} ... "
            f"{phonopy_vasp.config_name(len(runs))}"
        )

        _status(
            self.status,
            f"{len(runs)} folder{'' if len(runs) == 1 else 's'} written "
            f"to\n{self.root}\n{where}",
            theme.SUCCESS_STYLE,
        )

        # An INCAR chosen for the other route would carry the other
        # route's IBRION, which is the one tag that must match.
        self.incar = ""
        self.incar_label = ""

        self.kpoints_button.setEnabled(True)
        self.incar_button.setEnabled(True)
        self.generate_button.setEnabled(True)

        self._show_choices()

    def _ask_where(self) -> bool:
        """Pick the parent folder, once."""

        start = self.settings.value(WRITE_KEY, "", type=str)

        directory = QFileDialog.getExistingDirectory(
            self, "Save the run folders in", start
        )

        if not directory:
            return False

        self.settings.setValue(WRITE_KEY, directory)

        nx, ny, nz = self.supercell

        # Under one experiment folder rather than loose in whatever
        # directory was picked. A finite-displacement run is a hundred
        # directories; keeping them together is what lets
        # post-processing be handed the parent and still find them.
        self.root = (
            Path(directory)
            / phonopy_vasp.EXPERIMENT_DIR
            / f"{self.formula}_phonon_{nx}{ny}{nz}"
        )

        return True

    # ------------------------------------------------------------
    # Step three: fill them in
    # ------------------------------------------------------------

    def _generate(self) -> None:
        """Write INCAR, KPOINTS and POTCAR into the folders."""

        if self.config is None:
            return

        encut, encut_note = self._encut()

        potcar, potcar_problem = self._potcar()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            written = phonopy_vasp.write_vasp_inputs(
                self.config,
                self.phonon,
                encut=encut,
                kspacing=DEFAULT_KSPACING,
                potcar=potcar,
                scheme=self.scheme,
                incar=self.incar,
                mesh=self.mesh,
            )

        except Exception as exc:
            QApplication.restoreOverrideCursor()

            self._warn(
                "Could not write the inputs",
                f"{type(exc).__name__}: {exc}",
            )
            return

        finally:
            if QGuiApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        kx, ky, kz = self.config.mesh

        _status(
            self.status,
            f"{len(written)} files written.\n"
            f"K-mesh {kx} x {ky} x {kz}"
            + ("" if self.incar else f"   ENCUT {encut} eV"),
            theme.SUCCESS_STYLE,
        )

        if potcar_problem:
            self._warn("POTCAR", potcar_problem)

        # A hand-built INCAR carries its own ENCUT, so the derived one
        # was never used and there is nothing to apologise for.
        elif encut_note and not self.incar:
            self._warn("ENCUT", encut_note)

    # ------------------------------------------------------------
    # Derived numbers
    # ------------------------------------------------------------

    def _elements(self) -> list[str]:
        """Species of the unit cell, in POSCAR order."""

        return ordered_elements(
            phonopy_vasp.to_pymatgen(self.phonon.unitcell)
        )

    def _encut(self) -> tuple[int, str]:
        """
        ``(value, warning)`` for the cutoff, derived, not asked for.

        1.3 x the largest ENMAX of the POTCARs the structure would use.
        A phonon run wants the generous cutoff rather than the minimum:
        at the minimum the basis set is converged for the energy and not
        for its second derivative.
        """

        choices = library_choices(self.settings, self._elements())

        headed = [
            choice
            for choice in choices
            if choice.header is not None and choice.header.enmax is not None
        ]

        if not headed:
            return ENCUT_FALLBACK, (
                f"No pseudopotential library is set, so ENCUT could not "
                f"be derived from ENMAX. {ENCUT_FALLBACK} eV was "
                f"written. Set the library under "
                f"File > Set POTCAR Library and generate again."
            )

        enmax = max(choice.header.enmax for choice in headed)

        return (encut_for(enmax) or ENCUT_FALLBACK), ""

    def _potcar(self) -> tuple[str, str]:
        """``(text, problem)`` for the POTCAR that goes in every folder."""

        elements = self._elements()

        choices = library_choices(self.settings, elements)

        if not choices:
            return "", (
                f"No pseudopotential library is set, so no POTCAR was "
                f"written. Set it under File > Set POTCAR Library, "
                f"or copy one into each folder by hand -- the species "
                f"order is {' '.join(elements)}."
            )

        missing = [
            choice.element for choice in choices if choice.path is None
        ]

        if missing:
            return "", (
                f"The library has no usable POTCAR for "
                f"{', '.join(missing)}, so none was written."
            )

        try:
            return concatenate(choices), ""

        except (OSError, ValueError) as exc:
            return "", f"Could not assemble the POTCAR: {exc}"

    # ------------------------------------------------------------

    def _warn(self, title: str, text: str) -> None:
        """Everything that goes wrong arrives here."""

        QMessageBox.warning(self, title, text)


# ----------------------------------------------------------------
# Postprocessing
# ----------------------------------------------------------------


class VaspPostprocessDialog(QDialog):
    """
    Turn a finished configuration into a dispersion and a phonon DOS.

    Everything comes from the folder. It is inspected as soon as one is
    chosen, so "37 of 48 runs finished" is on screen before anything is
    started rather than after it fails.
    """

    def __init__(self, start: str = "", parent=None) -> None:

        super().__init__(parent)

        self.settings = QSettings(ORGANIZATION, APP_NAME)

        self.found = None

        self.setWindowTitle("Phonopy - VASP Post-Processing")
        fit_to_screen(self, 660, 520)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ----------------------------------------------------------

        row = QHBoxLayout()

        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("Configuration folder")
        self.root_edit.textChanged.connect(self._inspect)

        row.addWidget(self.root_edit, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        layout.addLayout(row)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)

        layout.addWidget(self.status)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setStyleSheet(theme.MONO_STYLE)
        self.detail.setLineWrapMode(QPlainTextEdit.NoWrap)

        layout.addWidget(self.detail, 1)

        # ----------------------------------------------------------

        layout.addWidget(self._build_options())

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.accept)

        buttons.addWidget(self.run_button)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        buttons.addWidget(close)

        layout.addLayout(buttons)

        # ----------------------------------------------------------

        self.root_edit.setText(
            start or self.settings.value(READ_KEY, "", type=str)
        )

        self._inspect()

    # ------------------------------------------------------------

    def _build_options(self) -> QWidget:
        """Sampling for the DOS and the band."""

        box = QGroupBox("Sampling")

        grid = QGridLayout(box)

        grid.addWidget(QLabel("DOS mesh"), 0, 0)

        self.mesh_spin = QSpinBox()
        self.mesh_spin.setRange(4, 80)
        self.mesh_spin.setValue(DEFAULT_MESH_N)
        self.mesh_spin.setSuffix(" per axis")

        grid.addWidget(self.mesh_spin, 0, 1)

        grid.addWidget(QLabel("Band points"), 1, 0)

        self.points_spin = QSpinBox()
        self.points_spin.setRange(11, 501)
        self.points_spin.setValue(DEFAULT_BAND_POINTS)
        self.points_spin.setSuffix(" per segment")

        grid.addWidget(self.points_spin, 1, 1)

        self.use_born = QCheckBox("Use born/ when it is there")
        self.use_born.setChecked(True)

        grid.addWidget(self.use_born, 2, 0, 1, 2)

        # Thermodynamics come off the same mesh as the DOS, so they cost
        # nothing extra once it has been sampled.
        self.want_thermal = QCheckBox(
            "Thermal properties (free energy, entropy, Cv)"
        )
        self.want_thermal.setChecked(True)
        self.want_thermal.toggled.connect(
            lambda on: self.t_max_spin.setEnabled(on)
        )

        grid.addWidget(self.want_thermal, 3, 0, 1, 2)

        grid.addWidget(QLabel("Temperature to"), 4, 0)

        self.t_max_spin = QSpinBox()
        self.t_max_spin.setRange(50, 5000)
        self.t_max_spin.setSingleStep(100)
        self.t_max_spin.setValue(1000)
        self.t_max_spin.setSuffix(" K")

        grid.addWidget(self.t_max_spin, 4, 1)

        grid.setColumnStretch(2, 1)

        return box

    # ------------------------------------------------------------

    def _browse(self) -> None:
        """Pick the configuration folder."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the phonon configuration folder",
            self.root_edit.text().strip()
            or self.settings.value(READ_KEY, "", type=str),
        )

        if directory:
            self.root_edit.setText(directory)

    def _inspect(self) -> None:
        """Look at whatever folder is currently named."""

        text = self.root_edit.text().strip()

        if not text:
            self.found = None
            _status(self.status, "Choose a folder.")
            self.detail.setPlainText("")
            self.run_button.setEnabled(False)
            return

        self.found = phonopy_vasp.discover(Path(text))

        if self.found.problems:
            style = theme.ERROR_STYLE

        elif self.found.missing:
            style = theme.WARNING_STYLE

        else:
            style = theme.SUCCESS_STYLE

        _status(self.status, self.found.summary(), style)

        self.detail.setPlainText(self._detail_text())

        self.run_button.setEnabled(self.found.usable)

    def _detail_text(self) -> str:
        """A listing of what was found, run by run."""

        found = self.found

        if found is None or found.problems:
            return ""

        nx, ny, nz = found.supercell

        lines = [
            f"Route      : {found.method}",
            f"Supercell  : {nx} x {ny} x {nz}   "
            f"({found.supercell_natoms} atoms)",
            f"Runs       : {len(found.found)} of {found.expected}",
            "",
        ]

        for path in found.found[:12]:
            lines.append(f"  ok       {path.parent.name}")

        if len(found.found) > 12:
            lines.append(f"  ...      and {len(found.found) - 12} more")

        for name in found.missing[:12]:
            lines.append(f"  waiting  {name}")

        if len(found.missing) > 12:
            lines.append(
                f"  ...      and {len(found.missing) - 12} more"
            )

        if found.born is not None:
            lines += ["", f"  born     {found.born.parent.name}"]

        return "\n".join(lines)

    # ------------------------------------------------------------

    def accept(self) -> None:
        """Remember the folder before handing back."""

        self.settings.setValue(READ_KEY, self.root_edit.text().strip())

        super().accept()

    def parameters(self) -> dict:
        """Everything :class:`~qvista.analyses.phonon_vasp.VaspPhononCollect` needs."""

        n = self.mesh_spin.value()

        # Whatever folder was typed, the configuration is whatever
        # discovery settled on inside it -- so an experiment folder or
        # its parent both hand the analysis the right directory.
        root = Path(self.root_edit.text().strip())

        if self.found is not None and not self.found.problems:
            root = self.found.root

        return {
            "root": root,
            "mesh": (n, n, n),
            "npoints": self.points_spin.value(),
            "use_born": self.use_born.isChecked(),
            "thermal": self.want_thermal.isChecked(),
            "t_max": float(self.t_max_spin.value()),
        }


# ----------------------------------------------------------------
# Guards, as message boxes
# ----------------------------------------------------------------


def confirm_size(natoms: int, parent=None) -> bool:
    """
    Ask before a supercell nobody meant to ask for.

    The one warning worth interrupting for: a repetition along a vacuum
    direction, or a large cell at low symmetry, turns one VASP run into
    a week of them, and the number is not obvious from three spin boxes.
    """

    if natoms <= SUPERCELL_ATOM_LIMIT:
        return True

    answer = QMessageBox.question(
        parent,
        "Large supercell",
        f"{natoms} atoms per run.\n\n"
        f"Above {SUPERCELL_ATOM_LIMIT} atoms each VASP run is a serious "
        f"calculation on its own. Continue?",
    )

    return answer == QMessageBox.Yes
