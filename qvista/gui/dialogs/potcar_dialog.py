"""
POTCAR assembly dialog.

Shows one row per element, stacked in POSCAR species order -- which is the
order the concatenated POTCAR will use -- with the variant that was
chosen for it, the cutoff and valence that variant carries, and where it
came from. The library directory is the user's own copy of VASP's
pseudopotentials; it is remembered between runs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import (
    ERROR_STYLE,
    INFO_BUTTON_STYLE,
    SUCCESS_BUTTON_STYLE,
    _clear_layout,
    _hline,
)
from qvista.services.potcar import (
    PBE_FAMILY,
    PotcarChoice,
    choice_for,
    concatenate,
    inspect_library,
    library_roots,
    recommended_encut,
    resolve_structure,
    total_valence,
    variant_candidates,
)

#: QSettings key under which the library directory is remembered.
LIBRARY_KEY = "potcar/library_root"

#: And which potential set inside it was chosen. Remembered separately
#: because a distribution can hold several (PBE, LDA, different
#: versions) and picking between them is a decision about the physics,
#: not something to re-derive from the folder each time.
POTENTIAL_SET_KEY = "potcar/potential_set"

#: Monospace, so the element rows line up under each other.
ROW_STYLE = theme.MONO_STYLE


def remembered_library(settings: QSettings) -> Path | None:
    """
    The potential set the user last chose, or ``None``.

    A distribution holds several sets side by side, so the remembered
    name is honoured when it is still there and the first set is taken
    otherwise -- never a set the user did not pick when the one they did
    is present.
    """

    stored = settings.value(LIBRARY_KEY, "", type=str)

    if not stored:
        return None

    try:
        sets = library_roots(Path(stored))

    except OSError:
        return None

    if not sets:
        return None

    remembered = settings.value(POTENTIAL_SET_KEY, "", type=str)

    for candidate in sets:
        if candidate.name == remembered:
            return candidate

    return sets[0]


def library_choices(
    settings: QSettings,
    elements: list[str],
    family: str = PBE_FAMILY,
) -> list[PotcarChoice]:
    """
    Resolve ``elements`` against the remembered library, quietly.

    Silent about failure on purpose: no library on a fresh install is
    normal, and every caller turns an empty result into a note beside
    whatever it was going to derive from ENMAX rather than an error.
    """

    if not elements:
        return []

    library = remembered_library(settings)

    if library is None:
        return []

    try:
        return resolve_structure(library, elements, family)

    except OSError:
        return []


class PotcarDialog(QDialog):
    """
    Assemble a POTCAR for the loaded structure.

    Each element gets a row showing the variant the family's rules picked,
    plus a dropdown to override it with any other variant the library
    actually has. Nothing is written until "Save POTCAR..." is clicked,
    and the dialog stays open so both families can be compared.
    """

    def __init__(
        self,
        elements: list[str],
        family: str,
        counts: dict[str, int] | None = None,
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.elements = list(elements)
        self.family = family
        self.counts = dict(counts or {})

        self.choices: list[PotcarChoice] = []
        self._overrides: dict[str, QComboBox] = {}

        #: Directory the user browsed to.
        self.root: Path | None = None

        #: The potential set inside it that is actually being read.
        self.library: Path | None = None

        #: Cached completeness check, per set. Scanning every variant on
        #: each dropdown change would stat the whole library each time.
        self._reports: dict[Path, object] = {}

        self.setWindowTitle(f"POTCAR ({family})")
        fit_to_screen(self, 720, 560)

        self.settings = QSettings(ORGANIZATION, APP_NAME)

        layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        # Library directory
        # ----------------------------------------------------------

        layout.addWidget(
            QLabel(
                f"Family: {family}    "
                f"Elements (POSCAR order): {', '.join(self.elements)}"
            )
        )

        root_row = QHBoxLayout()

        root_row.addWidget(QLabel("Library:"))

        self.root_label = QLabel()
        self.root_label.setWordWrap(True)

        root_row.addWidget(self.root_label, 1)

        browse_button = QPushButton("Browse...")
        browse_button.setStyleSheet(INFO_BUTTON_STYLE)
        browse_button.clicked.connect(self._browse)

        root_row.addWidget(browse_button)

        layout.addLayout(root_row)

        help_label = QLabel(
            "One sub-directory per potential (Li, Li_sv, Li_GW), or the "
            "folder holding the potpaw_* sets.\n"
            "VASP's pseudopotentials are licensed; QVista does not ship "
            "them."
        )
        help_label.setWordWrap(True)

        layout.addWidget(help_label)

        # A distribution can carry several sets side by side (PBE and LDA,
        # different versions). Which one to use is physics, so it is
        # offered rather than guessed; the row hides itself when there is
        # nothing to choose between.
        self.set_row = QWidget()

        set_layout = QHBoxLayout(self.set_row)
        set_layout.setContentsMargins(0, 0, 0, 0)

        set_layout.addWidget(QLabel("Potential set:"))

        self.set_box = QComboBox()
        self.set_box.currentIndexChanged.connect(self._on_set_changed)

        set_layout.addWidget(self.set_box, 1)

        layout.addWidget(self.set_row)

        layout.addWidget(_hline())

        # ----------------------------------------------------------
        # Per-element rows, stacked in POSCAR order
        # ----------------------------------------------------------

        rules = " → ".join(variant_candidates("El", family))

        layout.addWidget(QLabel(f"Preference order: {rules}"))

        self.rows_layout = QVBoxLayout()

        layout.addLayout(self.rows_layout)

        self.message = QLabel()
        self.message.setWordWrap(True)

        layout.addWidget(self.message)

        #: ENCUT and electron count implied by the chosen potentials.
        self.derived = QLabel()
        self.derived.setWordWrap(True)
        self.derived.setStyleSheet(ROW_STYLE)

        layout.addWidget(self.derived)

        layout.addStretch(1)

        # ----------------------------------------------------------
        # Buttons
        # ----------------------------------------------------------

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.save_button = QPushButton("Save POTCAR...")
        self.save_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)

        button_row.addWidget(self.save_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        button_row.addWidget(close_button)

        layout.addLayout(button_row)

        # ----------------------------------------------------------

        stored = self.settings.value(LIBRARY_KEY, "", type=str)

        self._set_root(Path(stored) if stored else None)

    # ------------------------------------------------------------

    def _set_root(self, root: Path | None) -> None:
        """Point at a library directory and re-resolve every element."""

        self.root = root

        self.root_label.setText(
            str(root) if root is not None else "(not set)"
        )

        sets = library_roots(root) if root is not None else []

        # Repopulating fires currentIndexChanged, which would re-enter
        # _refresh with a half-built list.
        self.set_box.blockSignals(True)
        self.set_box.clear()

        for candidate in sets:
            self.set_box.addItem(candidate.name, candidate)

        self.set_box.blockSignals(False)

        self.set_row.setVisible(len(sets) > 1)

        self.library = sets[0] if sets else None

        # Restore the set chosen last time, if it is still one of them.
        remembered = self.settings.value(POTENTIAL_SET_KEY, "", type=str)

        for index, candidate in enumerate(sets):

            if candidate.name == remembered:

                self.set_box.blockSignals(True)
                self.set_box.setCurrentIndex(index)
                self.set_box.blockSignals(False)

                self.library = candidate
                break

        self._refresh()

    def _on_set_changed(self, index: int) -> None:
        """Switch to another potential set within the same library."""

        if index < 0:
            return

        self.library = self.set_box.itemData(index)

        self._remember_set()

        self._refresh()

    def _browse(self) -> None:
        """Pick the library directory, and remember it."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the VASP pseudopotential library",
            str(self.root) if self.root is not None else "",
        )

        if not directory:
            return

        self.settings.setValue(LIBRARY_KEY, directory)

        self._set_root(Path(directory))

        self._remember_set()

    def _remember_set(self) -> None:
        """Persist the potential set now in use, for the next run."""

        if self.library is not None:
            self.settings.setValue(POTENTIAL_SET_KEY, self.library.name)

    def _refresh(self) -> None:
        """Rebuild the per-element rows for the current library."""

        _clear_layout(self.rows_layout)
        self._overrides = {}
        self.derived.setText("")

        if self.library is None:

            self.choices = []

            self.message.setStyleSheet(ERROR_STYLE)

            if self.root is None or not self.root.is_dir():
                self.message.setText(
                    "Set a library directory to see which potentials are "
                    "available."
                )
            else:
                self.message.setText(
                    f"No potentials under {self.root}. Expected element "
                    f"directories such as Li/POTCAR, either there or one "
                    f"level down in a potpaw_* set."
                )

            self.save_button.setEnabled(False)
            return

        self.choices = resolve_structure(
            self.library, self.elements, self.family
        )

        available = self._available_variants()

        for choice in self.choices:

            row = QHBoxLayout()

            label = QLabel(choice.status)
            label.setStyleSheet(
                ROW_STYLE + ("" if choice.found else f" {ERROR_STYLE}")
            )

            row.addWidget(label, 1)

            # Any variant of this element the library actually holds, so a
            # rule-picked default can be overridden without leaving the
            # dialog.
            options = available.get(choice.element, [])

            if options:

                box = QComboBox()
                box.addItems(options)

                if choice.found and choice.variant in options:
                    box.setCurrentText(choice.variant)

                box.currentTextChanged.connect(self._on_override)

                self._overrides[choice.element] = box

                row.addWidget(box)

            container = QWidget()
            container.setLayout(row)

            self.rows_layout.addWidget(container)

        self._update_state()

    def _available_variants(self) -> dict[str, list[str]]:
        """Every variant directory the library holds, per wanted element."""

        try:
            entries = sorted(
                entry.name
                for entry in self.library.iterdir()
                if entry.is_dir()
            )

        except OSError:
            return {}

        available: dict[str, list[str]] = {}

        for element in self.elements:

            # Compare the part before the first underscore, so "Li" picks
            # up "Li_sv" but never a different element's directory.
            available[element] = [
                name
                for name in entries
                if name.split("_")[0] == element
            ]

        return available

    def _on_override(self, _text: str) -> None:
        """Re-resolve after the user picked a different variant."""

        self._apply_overrides()
        self._update_state()

    def _apply_overrides(self) -> None:
        """Replace each choice with whatever its dropdown now names."""

        for index, choice in enumerate(self.choices):

            box = self._overrides.get(choice.element)

            if box is None:
                continue

            variant = box.currentText()

            if not variant or variant == choice.variant:
                continue

            # An override may name a variant outside the family's own
            # candidate list, so this is a direct lookup rather than a
            # re-run of the preference rules.
            self.choices[index] = choice_for(
                self.library, choice.element, variant, choice.tried
            )

    def _update_state(self) -> None:
        """Enable saving only when every element resolved."""

        missing = [
            choice.element for choice in self.choices if not choice.found
        ]

        if missing:

            unreadable = [
                choice.element
                for choice in self.choices
                if not choice.found and choice.unusable
            ]

            self.message.setStyleSheet(ERROR_STYLE)

            if unreadable:
                # Naming the elements is not enough here: every variant
                # in the dropdowns is equally empty, so the user works
                # through all of them and the button never lights up.
                # Say how much of the library is missing instead.
                self.message.setText(
                    f"No usable POTCAR for: {', '.join(missing)}. The "
                    f"directories for {', '.join(unreadable)} exist but "
                    f"hold empty files, and no choice in the dropdowns "
                    f"will fix that.\n"
                    f"{self._library_report().summary}"
                )
            else:
                self.message.setText(
                    f"No POTCAR found for: {', '.join(missing)}. "
                    f"Check the library directory, or pick a variant above."
                )

            self.derived.setText("")
            self.save_button.setEnabled(False)
            return

        order = " + ".join(
            f"{choice.variant}/POTCAR" for choice in self.choices
        )

        self.message.setStyleSheet("")
        self.message.setText(f"Will concatenate: {order}")

        self.derived.setText(self._derived_text())

        self.save_button.setEnabled(True)

    def _library_report(self):
        """Completeness of the current set, scanned once and kept."""

        if self.library not in self._reports:
            self._reports[self.library] = inspect_library(self.library)

        return self._reports[self.library]

    def _derived_text(self) -> str:
        """The run settings that follow from the chosen potentials."""

        lines: list[str] = []

        encut = recommended_encut(self.choices)

        if encut is not None:

            minimum, relax = encut

            lines.append(
                f"ENCUT ≥ {minimum} eV (highest ENMAX); "
                f"use {relax} eV for cell relaxation"
            )

        if self.counts:

            electrons = total_valence(self.choices, self.counts)

            if electrons is not None:

                formula = "  ".join(
                    f"{element}×{self.counts.get(element, 0)}"
                    for element in self.elements
                )

                lines.append(
                    f"Valence electrons in cell: {electrons}   ({formula})"
                )

        return "\n".join(lines)

    def _save(self) -> None:
        """Concatenate the chosen POTCARs and write them out."""

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save POTCAR",
            "POTCAR",
            "POTCAR File (POTCAR);;All Files (*)",
        )

        if not filename:
            return

        try:
            content = concatenate(self.choices)

            Path(filename).write_text(content, encoding="utf-8")

        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not write {filename}: {exc}"
            )
            return

        QMessageBox.information(
            self,
            "Saved",
            f"POTCAR written to {filename}\n\n"
            + "\n".join(
                f"{choice.element}: {choice.variant}"
                for choice in self.choices
            )
            + ("\n\n" + self._derived_text() if self._derived_text() else ""),
        )
