"""
Bader charge analysis of a finished VASP run.

Point the dialog at a folder holding the charge densities and it
partitions the cell into atomic volumes and reports what is inside each
one, three ways:

    All atoms       one row per atom: where it sits and the net charge
                    it carries, so the table answers "what is this atom's
                    charge" rather than "how many electrons are near it".

    Element wise    one row per species, carrying the mean net charge.

    Fragment wise   one row per group of atoms you choose, so the charge
                    transferred into or out of an adsorbate, a layer or a
                    polyanion is a single number rather than a sum you do
                    by hand.  Fragments are built first -- by cutting the
                    cell on an axis, by taking the rest, or by typing
                    atom numbers -- and the table follows.

Above the table is a panel saying what every column means, because a
Bader table is easy to misread: the CHARGE column is a population and
gets *larger* for an atom that has gained electrons, which is the
opposite sign to the charge everyone is actually after.

:mod:`qvista.services.bader` does the work and carries the references.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from qvista.core.config import APP_NAME, ORGANIZATION
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import SUCCESS_BUTTON_STYLE, _hline
from qvista.services import bader as bd

#: Where the run folder last opened is remembered between sessions.
RUN_KEY = "bader/run_root"

#: The three ways the table can be grouped.
SCOPES: dict[str, str] = {
    "all": "All atoms",
    "element": "Element wise",
    "fragment": "Fragment wise",
}

#: Decimal places for electron counts.  Four is what ACF.dat carries.
DECIMALS = 4

#: How tall the definitions panel opens, in pixels.  Enough for the
#: first few entries, and draggable from there.
DEFINITIONS_HEIGHT = 190

#: Room for the fragment table, in pixels -- about eight rows, which is
#: more than an axis cut plus a rest will ever produce.
FRAGMENT_TABLE_HEIGHT = 260

#: Lines of notes shown before the area starts scrolling. Whole lines,
#: never a fraction of one.
NOTES_LINES = 4

STATUS_STYLE = f"color: {theme.TEXT_DIM};"


def _status(label: QLabel, text: str, style: str = STATUS_STYLE) -> None:
    """Set the dialog's one line of feedback."""

    label.setStyleSheet(style)
    label.setText(text)


class Chip(QPushButton):
    """A small checkable box, used for the scope row."""

    def __init__(self, text: str, parent=None) -> None:

        super().__init__(text, parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(96)


class BaderRun(QThread):
    """
    One analysis, off the GUI thread.

    Partitioning a two-million-point grid takes the better part of a
    minute, which is far too long to hold the event loop: the window
    would stop repainting and Windows would offer to close it.
    """

    #: ``(message, percent)``.
    progress = Signal(str, int)

    #: The finished :class:`~qvista.services.bader.BaderResult`.
    completed = Signal(object)

    #: A message written for the user.
    failed = Signal(str)

    def __init__(
        self,
        folder: str,
        reference: str,
        method: str,
        vacuum: bool,
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.folder = folder
        self.reference = reference
        self.method = method
        self.vacuum = vacuum

    def run(self) -> None:

        try:
            result = bd.analyse(
                self.folder,
                reference=self.reference,
                method=self.method,
                vacuum=self.vacuum,
                progress=self.progress.emit,
            )

        except bd.BaderError as exc:
            self.failed.emit(str(exc))
            return

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        self.completed.emit(result)


class BaderDialog(QDialog):
    """The window behind Analysis > Bonding Tools > Bader Analysis."""

    def __init__(self, start: str = "", parent=None):

        super().__init__(parent)

        self.result: bd.BaderResult | None = None
        self.scope = "all"
        self.worker: BaderRun | None = None

        #: The span the cutting plane was last offered against, so that a
        #: chosen plane survives a trip out to the table and back.
        self._cut_span: tuple[float, float] | None = None

        self.setWindowTitle("Bader Analysis")
        fit_to_screen(self, 1120, 880)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._build_source(layout)

        layout.addWidget(_hline())

        # A splitter rather than two fixed panes: the definitions are
        # worth a lot of room the first time and none of it afterwards,
        # and which of those a given user is doing is not knowable here.
        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)

        self.split.addWidget(self._build_definitions())
        self.split.addWidget(self._build_results())

        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([DEFINITIONS_HEIGHT, 640])

        layout.addWidget(self.split, 1)

        # Folded away to start with. Open, it takes a fifth of the
        # dialog and still ends mid-sentence, because there is more text
        # than any pane height happens to land on; the toggle says what
        # it holds and one click brings it back.
        self.definitions_toggle.setChecked(False)

        self.status = QLabel("Choose the folder holding the VASP run.")
        _status(self.status, self.status.text())
        self.status.setWordWrap(True)

        layout.addWidget(self.status)

        footer = QHBoxLayout()
        footer.addStretch(1)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        footer.addWidget(close)

        layout.addLayout(footer)

        if start:
            self.folder.setText(start)
            self._offer_references()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_source(self, layout: QVBoxLayout) -> None:
        """Folder, what to partition, and how."""

        row = QHBoxLayout()

        row.addWidget(QLabel("Run folder"))

        self.folder = QLineEdit()
        self.folder.setPlaceholderText(
            "the directory holding CHGCAR, AECCAR0, AECCAR2 and POTCAR"
        )
        self.folder.editingFinished.connect(self._offer_references)

        row.addWidget(self.folder, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)

        row.addWidget(browse)

        layout.addLayout(row)

        options = QHBoxLayout()
        options.setSpacing(10)

        options.addWidget(QLabel("Reference"))

        self.reference = QComboBox()
        self.reference.setMinimumWidth(230)
        self.reference.currentIndexChanged.connect(self._describe_choices)

        options.addWidget(self.reference)

        options.addWidget(QLabel("Method"))

        self.method = QComboBox()
        self.method.setMinimumWidth(140)

        for key in bd.METHODS:
            self.method.addItem(key, key)

        self.method.currentIndexChanged.connect(self._describe_choices)

        options.addWidget(self.method)

        # Off by default: in a dense solid there is no vacuum to find and
        # the option only costs a pass over the grid.
        self.vacuum = QCheckBox("Set vacuum aside")
        self.vacuum.setToolTip(
            "Grid points below "
            f"{bd.VACUUM_TOLERANCE:g} e/A^3 are counted as vacuum instead "
            "of being given to an atom. For slabs and molecules in a box."
        )

        options.addWidget(self.vacuum)

        options.addStretch(1)

        self.run_button = QPushButton("Run analysis")
        self.run_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.run_button.clicked.connect(self._run)

        options.addWidget(self.run_button)

        layout.addLayout(options)

        self.choice_note = QLabel("")
        self.choice_note.setStyleSheet(STATUS_STYLE)
        self.choice_note.setWordWrap(True)

        layout.addWidget(self.choice_note)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)

        layout.addWidget(self.progress)

    def _build_definitions(self) -> QWidget:
        """
        What every column means, above the table rather than beside it.

        A Bader table is genuinely easy to misread, so the wording comes
        from the service, where it sits next to the code that produces
        the numbers and can be tested.
        """

        box = QWidget()

        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        header = QHBoxLayout()

        self.definitions_toggle = QPushButton("What these numbers mean")
        self.definitions_toggle.setCheckable(True)
        self.definitions_toggle.setChecked(True)
        self.definitions_toggle.setCursor(Qt.PointingHandCursor)
        self.definitions_toggle.toggled.connect(self._toggle_definitions)

        header.addWidget(self.definitions_toggle)
        header.addStretch(1)

        column.addLayout(header)

        # No maximum height: the splitter above decides how tall this
        # starts, and capping it here would stop it being dragged open.
        self.definitions = QTextBrowser()
        self.definitions.setOpenExternalLinks(True)
        self.definitions.setMinimumHeight(90)
        self.definitions.setHtml(self._definitions_html())

        column.addWidget(self.definitions)

        return box

    def _definitions_html(self) -> str:
        """The service's wording, as something a browser can lay out."""

        parts = [
            f"<style>body{{color:{theme.TEXT};}} "
            f"dt{{color:{theme.IRIS_GLOW};font-weight:600;margin-top:6px;}} "
            f"dd{{color:{theme.TEXT_DIM};margin-left:14px;}}</style>",
            "<dl>",
        ]

        for term, text in bd.DEFINITIONS:
            parts.append(f"<dt>{term}</dt><dd>{text}</dd>")

        parts.append("</dl>")

        parts.append(
            f"<p style='color:{theme.TEXT_FAINT};margin-top:10px;'>"
            "<b>Where this comes from</b></p><ul>"
        )

        for title, url in bd.SOURCES:

            if url:
                parts.append(
                    f"<li style='color:{theme.TEXT_DIM};'>"
                    f"<a href='{url}' style='color:{theme.AQUA};'>"
                    f"{title}</a></li>"
                )
            else:
                parts.append(
                    f"<li style='color:{theme.TEXT_DIM};'>{title}</li>"
                )

        parts.append("</ul>")

        return "".join(parts)

    def _toggle_definitions(self, shown: bool) -> None:
        """
        Hide the panel and give its room to the table.

        Hiding the browser alone is not enough: the splitter keeps the
        space it was holding, so collapsing the definitions would leave
        an empty band where they used to be.
        """

        self.definitions.setVisible(shown)

        self.definitions_toggle.setText(
            "What these numbers mean"
            if shown
            else "What these numbers mean  (show)"
        )

        split = getattr(self, "split", None)

        if split is None:
            return

        total = sum(split.sizes())

        top = (
            min(DEFINITIONS_HEIGHT, total // 3)
            if shown
            else self.definitions_toggle.sizeHint().height() + 8
        )

        split.setSizes([top, max(1, total - top)])

    def _build_results(self) -> QGroupBox:
        """Scope row, the fragment editor, and the table."""

        box = QGroupBox("Charges")

        layout = QVBoxLayout(box)

        scope_row = QHBoxLayout()
        scope_row.setSpacing(6)

        scope_row.addWidget(QLabel("Show"))

        self.scope_chips: dict[str, Chip] = {}

        for key, text in SCOPES.items():

            chip = Chip(text)
            chip.setChecked(key == "all")
            chip.clicked.connect(
                lambda _=False, name=key: self._pick_scope(name)
            )

            self.scope_chips[key] = chip

            scope_row.addWidget(chip)

        scope_row.addStretch(1)

        self.summary = QLabel("")
        self.summary.setStyleSheet(STATUS_STYLE)

        scope_row.addWidget(self.summary)

        layout.addLayout(scope_row)

        # Stretches, because while fragments are being chosen the builder
        # is the whole of the panel -- the table it replaces is empty
        # until they exist.
        layout.addWidget(self._build_fragments(), 1)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        layout.addWidget(self.table, 1)

        # Bounded and scrolled: the notes are worth reading but there can
        # be five of them, and left to grow they take the table's room.
        self.notes = QLabel("")
        self.notes.setStyleSheet(theme.WARNING_STYLE)
        self.notes.setWordWrap(True)
        self.notes.setAlignment(Qt.AlignTop)

        self.notes_area = QScrollArea()
        self.notes_area.setWidgetResizable(True)
        self.notes_area.setWidget(self.notes)
        self.notes_area.setFrameShape(QFrame.NoFrame)
        self.notes_area.setVisible(False)

        # A whole number of lines. Left to an arbitrary pixel height the
        # area ends mid-glyph, and a half-rendered line of text reads as
        # two lines printed on top of each other.
        self.notes_area.setMaximumHeight(
            NOTES_LINES * self.notes.fontMetrics().lineSpacing()
            + 2 * self.notes_area.frameWidth()
        )

        layout.addWidget(self.notes_area)

        buttons = QHBoxLayout()

        # Only up while a fragment table is showing: it is the way back
        # to the builder the table came from.
        self.edit_button = QPushButton("Edit fragments...")
        self.edit_button.setVisible(False)
        self.edit_button.clicked.connect(self._edit_fragments)

        buttons.addWidget(self.edit_button)

        buttons.addStretch(1)

        self.save_button = QPushButton("Save table...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_table)

        buttons.addWidget(self.save_button)

        self.visualize = QPushButton("Visualization...")
        self.visualize.setEnabled(False)
        self.visualize.clicked.connect(self._visualize)

        buttons.addWidget(self.visualize)

        layout.addLayout(buttons)

        return box

    def _build_fragments(self) -> QWidget:
        """
        The fragment builder, which stands in place of the table.

        Fragments are chosen before there is anything to show, so the
        builder gets the dialog to itself rather than a strip above a
        table that stays empty until the fragments exist.  Three ways in:
        cut the cell on an axis, take whatever is not yet spoken for, or
        type atom numbers.  The last is the general case -- ``7-42`` is a
        great deal quicker than ticking thirty-six boxes -- and it is the
        same syntax the visualization window takes.
        """

        self.fragment_box = QGroupBox("1. Choose the fragments")

        column = QVBoxLayout(self.fragment_box)

        # -- cut the cell in two on one axis ---------------------
        cut_row = QHBoxLayout()

        cut_row.addWidget(QLabel("Cut on"))

        self.cut_axis = QComboBox()

        for index, name in enumerate("XYZ"):
            self.cut_axis.addItem(name, index)

        self.cut_axis.setCurrentIndex(2)
        self.cut_axis.currentIndexChanged.connect(self._describe_axis)

        cut_row.addWidget(self.cut_axis)

        cut_row.addWidget(QLabel("at"))

        self.cut_value = QDoubleSpinBox()
        self.cut_value.setDecimals(2)
        self.cut_value.setRange(-9999.0, 9999.0)
        self.cut_value.setSingleStep(0.5)
        self.cut_value.setSuffix(" A")

        cut_row.addWidget(self.cut_value)

        split = QPushButton("Split")
        split.setToolTip(
            "Add two fragments: the atoms below the plane, and the "
            "atoms above it."
        )
        split.clicked.connect(self._split_on_axis)

        cut_row.addWidget(split)

        self.axis_note = QLabel("")
        self.axis_note.setStyleSheet(STATUS_STYLE)

        cut_row.addWidget(self.axis_note, 1)

        column.addLayout(cut_row)

        hint = QLabel(
            "One fragment per row: a name, and the atoms in it as numbers "
            "or ranges such as 1-6 or 3,7,11. Atoms may appear in more "
            "than one fragment, or in none."
        )
        hint.setStyleSheet(STATUS_STYLE)
        hint.setWordWrap(True)

        column.addWidget(hint)

        self.fragments = QTableWidget(0, 2)
        self.fragments.setHorizontalHeaderLabels(["Name", "Atoms"])
        self.fragments.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.fragments.verticalHeader().setVisible(False)
        self.fragments.setMinimumHeight(FRAGMENT_TABLE_HEIGHT)

        column.addWidget(self.fragments, 1)

        row = QHBoxLayout()

        add = QPushButton("Add fragment")
        add.clicked.connect(lambda: self._add_fragment())

        row.addWidget(add)

        rest = QPushButton("Rest")
        rest.setToolTip(
            "Add one fragment holding every atom that is not in a "
            "fragment already."
        )
        rest.clicked.connect(self._add_rest)

        row.addWidget(rest)

        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_fragment)

        row.addWidget(remove)

        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_fragments)

        row.addWidget(clear)

        row.addStretch(1)

        self.show_table = QPushButton("Show table")
        self.show_table.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.show_table.clicked.connect(self._show_fragment_table)

        row.addWidget(self.show_table)

        column.addLayout(row)

        self.fragment_box.setVisible(False)

        return self.fragment_box

    # ------------------------------------------------------------
    # Choosing what to run
    # ------------------------------------------------------------

    def _browse(self) -> None:
        """Pick the run folder, starting where one was last opened."""

        settings = QSettings(ORGANIZATION, APP_NAME)

        start = self.folder.text() or settings.value(RUN_KEY, "", type=str)

        chosen = QFileDialog.getExistingDirectory(
            self, "VASP run folder", start
        )

        if chosen:
            self.folder.setText(chosen)
            settings.setValue(RUN_KEY, chosen)
            self._offer_references()

    def _offer_references(self) -> None:
        """
        Offer only the references the folder has the files for.

        Listing the all-electron option for a run made without
        ``LAECHG = .TRUE.`` would put the right answer in front of the
        user and then fail on it.
        """

        folder = self.folder.text().strip()

        self.reference.clear()

        if not folder or not Path(folder).is_dir():
            _status(self.status, "That folder does not exist.", theme.WARNING_STYLE)
            return

        found = bd.available_references(folder)

        if not found:
            _status(
                self.status,
                "No CHGCAR in that folder. Bader analysis needs the charge "
                "density, which VASP writes when LCHARG = .TRUE.",
                theme.WARNING_STYLE,
            )
            return

        for key in found:
            self.reference.addItem(key, key)

        if "all-electron" not in found:
            _status(
                self.status,
                "No AECCAR0/AECCAR2 in that folder, so only the valence "
                "density is available. Rerun with LAECHG = .TRUE. for "
                "boundaries drawn from the all-electron density.",
                theme.WARNING_STYLE,
            )
        else:
            _status(self.status, "Ready.")

        self._describe_choices()

    def _describe_choices(self) -> None:
        """Say what the two combo boxes currently mean."""

        reference = self.reference.currentData()
        method = self.method.currentData()

        if reference is None or method is None:
            return

        self.choice_note.setText(
            f"{bd.REFERENCES[reference]}  --  {bd.METHODS[method]}"
        )

    def _pick_scope(self, name: str) -> None:
        """Group the table by atom, element or fragment."""

        self.scope = name

        for key, chip in self.scope_chips.items():
            chip.setChecked(key == name)

        if name != "fragment":
            self._show_builder(False)
            self._refresh_table()
            return

        if self.fragments.rowCount() == 0:
            self._seed_fragments()

        # Fragments are chosen first and the table follows, so entering
        # this scope lands on the builder rather than on a table of
        # whatever happened to be in the editor last time.
        self._describe_axis()
        self._show_builder(True)

    def _show_builder(self, building: bool) -> None:
        """Swap between choosing fragments and reading the table."""

        self.fragment_box.setVisible(building)
        self.table.setVisible(not building)

        # The notes are about how the density was partitioned, which is
        # nothing to do with picking atoms. They belong beside the table
        # they qualify, not over the builder.
        self._show_notes()

        if building:
            self._give_builder_room()

        self.edit_button.setVisible(
            not building and self.scope == "fragment"
        )

        self.save_button.setEnabled(
            not building and self.result is not None
        )
        self.visualize.setEnabled(
            not building and self.result is not None
        )

    def _show_notes(self) -> None:
        """Notes belong with the table, and only when there are any."""

        self.notes_area.setVisible(
            bool(self.notes.text())
            and not self.fragment_box.isVisible()
        )

    def _give_builder_room(self) -> None:
        """
        Let the fragment table stand at its full height.

        The splitter keeps whatever sizes it was last given, which for
        the builder is usually a few rows short, so the definitions --
        the compressible half of the window -- give the difference up.
        The pane stays open at its minimum rather than closing, so
        dragging it back is one gesture.
        """

        split = getattr(self, "split", None)

        if split is None:
            return

        wanted = self.fragment_box.sizeHint().height()

        sizes = split.sizes()
        total = sum(sizes)

        if total <= 0 or len(sizes) < 2 or sizes[1] >= wanted:
            return

        floor = self.definitions_toggle.sizeHint().height() + 8

        if self.definitions.isVisible():
            floor += self.definitions.minimumHeight()

        top = max(floor, total - wanted)

        split.setSizes([top, max(1, total - top)])

    def _edit_fragments(self) -> None:
        """Back to the builder from the table it produced."""

        self._describe_axis()
        self._show_builder(True)

    def _show_fragment_table(self) -> None:
        """Leave the builder, if what it holds makes a table."""

        if not self._read_fragments():
            _status(
                self.status,
                "No fragments yet. Cut the cell on an axis, or add a row "
                "and type the atom numbers in it.",
                theme.WARNING_STYLE,
            )
            return

        self._show_builder(False)
        self._refresh_table()

    # ------------------------------------------------------------
    # Running
    # ------------------------------------------------------------

    def _run(self) -> None:
        """Start the analysis on its own thread."""

        folder = self.folder.text().strip()

        if not folder:
            QMessageBox.information(
                self,
                "No folder",
                "Choose the folder holding the VASP run first.",
            )
            return

        if self.reference.currentData() is None:
            self._offer_references()

            if self.reference.currentData() is None:
                return

        if self.worker is not None and self.worker.isRunning():
            return

        self.run_button.setEnabled(False)
        self.visualize.setEnabled(False)
        self.save_button.setEnabled(False)

        self.progress.setValue(0)
        self.progress.setVisible(True)

        self.worker = BaderRun(
            folder,
            self.reference.currentData(),
            self.method.currentData(),
            self.vacuum.isChecked(),
            parent=self,
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.worker.start()

    def _on_progress(self, message: str, percent: int) -> None:

        _status(self.status, message)

        if percent >= 0:
            self.progress.setValue(percent)

    def _on_completed(self, result: bd.BaderResult) -> None:

        self.result = result

        self._build_summary()

        self.notes.setText("\n".join(f"- {note}" for note in result.notes))
        self._show_notes()

        self._refresh_table()

        # The axis spans come from the structure, which only exists now.
        self._describe_axis()

        # Not while the fragment builder is up: there is no table on
        # screen yet to save or to draw.
        building = self.fragment_box.isVisible()

        self.save_button.setEnabled(not building)
        self.visualize.setEnabled(not building)

        _status(
            self.status,
            f"Done. {result.electrons:.4f} electrons over "
            f"{len(result.atoms)} atoms.",
            theme.SUCCESS_STYLE,
        )

    def _on_failed(self, message: str) -> None:

        _status(self.status, message, theme.ERROR_STYLE)

        QMessageBox.warning(self, "Bader Analysis", message)

    def _on_finished(self) -> None:

        self.progress.setVisible(False)
        self.run_button.setEnabled(True)

        self.worker = None

    def _build_summary(self) -> None:

        if self.result is None:
            return

        result = self.result

        text = (
            f"{result.reference}, {result.method}, "
            f"{result.grid[0]}x{result.grid[1]}x{result.grid[2]} grid"
        )

        if result.vacuum_volume:
            text += (
                f", vacuum {result.vacuum_charge:.3f} e / "
                f"{result.vacuum_volume:.1f} A^3"
            )

        self.summary.setText(text)

    # ------------------------------------------------------------
    # Fragments
    # ------------------------------------------------------------

    def _seed_fragments(self) -> None:
        """
        Start the editor off with one fragment per element.

        Something meaningful on screen beats an empty grid: it shows the
        syntax, it is a real answer on its own, and it is quicker to edit
        than to write from nothing.
        """

        if self.result is None:
            self._add_fragment("Fragment 1", "")
            return

        for row in bd.by_element(self.result):

            members = list(row.members)

            self._add_fragment(row.label, _compress(members))

    def _describe_axis(self) -> None:
        """
        Say how far the cell runs along the chosen axis.

        A cutting plane is only choosable against the span it is cutting:
        "5 A" means nothing until you know z runs from 0 to 14.
        """

        if self.result is None:
            self.axis_note.setText("")
            return

        axis = self.cut_axis.currentData()

        low, high = bd.axis_span(self.result, axis)

        self.axis_note.setText(
            f"{bd.AXES[axis]} runs {low:.2f} to {high:.2f} A over "
            f"{len(self.result.atoms)} atoms"
        )

        self.cut_value.setRange(low, high)

        # Halfway is the useful first guess, and it is always a cut that
        # puts atoms on both sides.  It is offered whenever the span
        # itself changes -- a new structure, or a different axis, where
        # whatever was typed for the old one no longer means anything --
        # but not on the way back from the table, which would throw away
        # a plane the user chose.
        span = (round(low, 6), round(high, 6))

        if span != self._cut_span:
            self._cut_span = span
            self.cut_value.setValue((low + high) / 2.0)

    def _split_on_axis(self) -> None:
        """Two fragments: what lies below the plane, and what lies above."""

        if self.result is None:
            return

        axis = self.cut_axis.currentData()
        name = bd.AXES[axis]
        cut = self.cut_value.value()

        below, above = bd.split_on_axis(self.result, axis, cut)

        if not below or not above:
            _status(
                self.status,
                f"Every atom is on one side of {name} = {cut:.2f} A, so "
                "that plane does not split anything.",
                theme.WARNING_STYLE,
            )
            return

        self._add_fragment(f"{name} below {cut:.2f}", _compress(below))
        self._add_fragment(f"{name} above {cut:.2f}", _compress(above))

        _status(
            self.status,
            f"{name} = {cut:.2f} A splits the cell {len(below)} / "
            f"{len(above)}.",
        )

    def _add_rest(self) -> None:
        """One fragment holding every atom not already in one."""

        if self.result is None:
            return

        try:
            rest = bd.remaining(self.result, self._read_fragments())

        except bd.BaderError as exc:
            _status(self.status, str(exc), theme.ERROR_STYLE)
            return

        if not rest:
            _status(
                self.status,
                "Every atom is in a fragment already, so there is no "
                "rest to take.",
                theme.WARNING_STYLE,
            )
            return

        self._add_fragment("Rest", _compress(rest))

        _status(self.status, f"Rest holds {len(rest)} atoms.")

    def _clear_fragments(self) -> None:

        self.fragments.setRowCount(0)

    def _add_fragment(self, name: str = "", atoms: str = "") -> None:

        row = self.fragments.rowCount()

        self.fragments.insertRow(row)
        self.fragments.setItem(
            row, 0, QTableWidgetItem(name or f"Fragment {row + 1}")
        )
        self.fragments.setItem(row, 1, QTableWidgetItem(atoms))

    def _remove_fragment(self) -> None:

        row = self.fragments.currentRow()

        if row < 0:
            row = self.fragments.rowCount() - 1

        if row >= 0:
            self.fragments.removeRow(row)

    def _read_fragments(self) -> dict[str, list[int]]:
        """What the editor currently says, as the service wants it."""

        if self.result is None:
            return {}

        count = len(self.result.atoms)

        fragments: dict[str, list[int]] = {}

        for row in range(self.fragments.rowCount()):

            name_item = self.fragments.item(row, 0)
            atoms_item = self.fragments.item(row, 1)

            text = atoms_item.text().strip() if atoms_item else ""

            if not text:
                continue

            name = (
                name_item.text().strip()
                if name_item and name_item.text().strip()
                else f"Fragment {row + 1}"
            )

            # Two fragments with the same name would collapse into one
            # row and silently lose the other.
            while name in fragments:
                name += "'"

            fragments[name] = bd.parse_selection(text, count)

        return fragments

    # ------------------------------------------------------------
    # The table
    # ------------------------------------------------------------

    def _refresh_table(self) -> None:

        if self.result is None:
            return

        try:
            if self.scope == "all":
                self._fill_atoms()

            elif self.scope == "element":
                self._fill_groups(
                    bd.by_element(self.result), "Element", full=False
                )

            else:
                fragments = self._read_fragments()

                if not fragments:
                    self.table.clear()
                    self.table.setRowCount(0)
                    self.table.setColumnCount(0)
                    return

                self._fill_groups(
                    bd.by_fragment(self.result, fragments), "Fragment"
                )

        except bd.BaderError as exc:
            _status(self.status, str(exc), theme.ERROR_STYLE)

    def _fill_atoms(self) -> None:
        """
        One row per atom: where it sits, and the charge it carries.

        The population, ZVAL and volume behind the net charge are all
        intermediate steps, and putting them in the table invites the
        commonest misreading of a Bader result -- quoting the population,
        which runs the opposite way to the charge.
        """

        atoms = self.result.atoms

        columns = ["Atom", "X", "Y", "Z", "Net charge"]

        self.table.clear()
        self.table.setRowCount(len(atoms) + 1)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        for row, atom in enumerate(atoms):

            self._set(row, 0, atom.label())

            for offset, value in enumerate(atom.coords):
                self._set(row, 1 + offset, f"{float(value):.4f}")

            self._set(row, 4, f"{atom.net:+.{DECIMALS}f}")

        last = len(atoms)

        self._set(last, 0, "Total", bold=True)

        for column in (1, 2, 3):
            self._set(last, column, "", bold=True)

        self._set(
            last,
            4,
            f"{sum(a.net for a in atoms):+.{DECIMALS}f}",
            bold=True,
        )

    def _fill_groups(
        self,
        rows: list[bd.GroupCharge],
        first: str,
        full: bool = True,
    ) -> None:
        """
        One row per element or per fragment.

        ``full`` decides how much of the arithmetic is shown.  Element
        wise, the mean net charge is the number that gets quoted and the
        populations behind it only crowd it out.  A fragment is the
        opposite case: naming a group of atoms is usually a way of asking
        what the group as a whole holds, so its totals stay.
        """

        if full:
            columns = [
                first,
                "Atoms",
                "Charge",
                "Charge/atom",
                "ZVAL",
                "Net charge",
                "Net/atom",
                "Volume",
                "Volume/atom",
            ]
        else:
            columns = [first, "Atoms", "Net/atom"]

        self.table.clear()
        self.table.setRowCount(len(rows) + 1)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        for row, entry in enumerate(rows):

            self._set(row, 0, entry.label)
            self._set(row, 1, str(entry.count))

            if not full:
                self._set(row, 2, f"{entry.net_per_atom:+.{DECIMALS}f}")
                continue

            self._set(row, 2, f"{entry.charge:.{DECIMALS}f}")
            self._set(row, 3, f"{entry.charge_per_atom:.{DECIMALS}f}")
            self._set(row, 4, f"{entry.zval:.2f}")
            self._set(row, 5, f"{entry.net:+.{DECIMALS}f}")
            self._set(row, 6, f"{entry.net_per_atom:+.{DECIMALS}f}")
            self._set(row, 7, f"{entry.volume:.4f}")
            self._set(row, 8, f"{entry.volume_per_atom:.4f}")

        last = len(rows)

        self._set(last, 0, "Total", bold=True)
        self._set(last, 1, str(sum(e.count for e in rows)), bold=True)

        if not full:
            # An average of averages is not the cell's average, and the
            # cell's is zero by construction anyway.
            self._set(last, 2, "", bold=True)
            return

        self._set(
            last,
            2,
            f"{sum(e.charge for e in rows):.{DECIMALS}f}",
            bold=True,
        )
        self._set(last, 3, "", bold=True)
        self._set(last, 4, f"{sum(e.zval for e in rows):.2f}", bold=True)
        self._set(
            last,
            5,
            f"{sum(e.net for e in rows):+.{DECIMALS}f}",
            bold=True,
        )
        self._set(last, 6, "", bold=True)
        self._set(
            last, 7, f"{sum(e.volume for e in rows):.4f}", bold=True
        )
        self._set(last, 8, "", bold=True)

    def _set(
        self,
        row: int,
        column: int,
        text: str,
        bold: bool = False,
    ) -> None:
        """Place one read-only cell."""

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

        if column > 0:
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)

        self.table.setItem(row, column, item)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    def _save_table(self) -> None:
        """Write what is on screen as CSV, with the run described on top."""

        if self.result is None:
            return

        suggested = f"bader_{self.scope}.csv"

        start = str(Path(self.folder.text().strip()) / suggested)

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save table", start, "CSV (*.csv)"
        )

        if not chosen:
            return

        headers = [
            self.table.horizontalHeaderItem(column).text()
            for column in range(self.table.columnCount())
        ]

        try:
            with open(chosen, "w", newline="", encoding="utf-8") as handle:

                writer = csv.writer(handle)

                writer.writerow(["# Bader analysis"])
                writer.writerow([f"# reference {self.result.reference}"])
                writer.writerow([f"# method {self.result.method}"])
                writer.writerow(
                    [
                        "# grid "
                        f"{self.result.grid[0]}x{self.result.grid[1]}"
                        f"x{self.result.grid[2]}"
                    ]
                )
                writer.writerow(
                    [f"# electrons {self.result.electrons:.6f}"]
                )

                for note in self.result.notes:
                    writer.writerow([f"# {note}"])

                writer.writerow(headers)

                for row in range(self.table.rowCount()):
                    writer.writerow(
                        [
                            (
                                self.table.item(row, column).text()
                                if self.table.item(row, column)
                                else ""
                            )
                            for column in range(self.table.columnCount())
                        ]
                    )

        except OSError as exc:
            QMessageBox.warning(self, "Save table", str(exc))
            return

        _status(self.status, f"Saved {Path(chosen).name}")

    def _visualize(self) -> None:
        """Open the charges over the structure, in their own window."""

        from qvista.gui.dialogs.bader_viewer import BaderViewer

        if self.result is None:
            return

        try:
            viewer = BaderViewer(
                self.result,
                heading=(
                    f"Bader   {self.result.reference}   "
                    f"{self.result.method}   "
                    f"{len(self.result.atoms)} atoms"
                ),
                parent=self,
            )

        except Exception as exc:  # noqa: BLE001 - shown, not swallowed
            _status(self.status, str(exc), theme.ERROR_STYLE)
            QMessageBox.critical(self, "Visualization", str(exc))
            return

        viewer.show()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt name

        # The thread holds a reference to this dialog as its parent, so
        # letting it outlive the window would crash on the next signal.
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(15000)

        super().closeEvent(event)


def _compress(numbers: list[int]) -> str:
    """``[1,2,3,7]`` as ``"1-3,7"``, which is what the box expects back."""

    if not numbers:
        return ""

    ordered = sorted(numbers)

    pieces: list[str] = []

    start = previous = ordered[0]

    for value in ordered[1:] + [None]:

        if value is not None and value == previous + 1:
            previous = value
            continue

        pieces.append(
            str(start) if start == previous else f"{start}-{previous}"
        )

        if value is not None:
            start = previous = value

    return ",".join(pieces)
