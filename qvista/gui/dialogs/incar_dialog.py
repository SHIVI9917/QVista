"""
INCAR generation dialog.

Eight boxes across the top, one per main job. Clicking one is most of the
setup: it brings that job's settings, each with its legal values and what
they mean, and the INCAR appears beside them as it is assembled.

Under the boxes sit the choices that only some jobs have. An AIMD run
asks which ensemble and then which thermostat, and each answer reveals
the settings that answer needs -- there is no point offering SMASS to
someone who picked Langevin. A phonon asks DFPT or finite differences. A
single point is the exception: its sub-jobs are not exclusive, so a
static SCF and a DOS and a Bader run can all be ticked, and each one
ticked adds to the same file rather than starting another.

Then the optional physics -- a Hubbard U, a hybrid, dispersion,
spin-orbit, a meta-GGA, implicit solvation -- none of it on until asked
for, and last the settings that have nothing to do with the science:
what to keep on disk, how to split the work, how to mix the density.

The file on the right is the whole point. A tag and its effect are never
more than a glance apart, and because the assembly rule lets a later
block override an earlier one, the only way to be sure what LWAVE ends
up as is to look -- so the file is always there to look at.

Nothing on screen explains itself in prose. A heading, then the tags
under it as a name and an editor, and that is all: two hundred settings
each captioned with a sentence is a wall to read past on the way to the
one being changed. What every tag does is still there on hover, where it
costs nothing to ignore.

Everything about what the tags *are* lives in :mod:`qvista.services.incar`.
This module only lays them out.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import (
    INFO_BUTTON_STYLE,
    SUCCESS_BUTTON_STYLE,
    _hline,
)
from qvista.gui.widgets.flow_layout import flow_widget
from qvista.services import incar as ic

#: The preview, and anything that has to line up with it.
MONO_STYLE = theme.MONO_STYLE

#: The heading over a run of settings.
SECTION_STYLE = (
    f"color: {theme.IRIS_GLOW}; font-weight: 700; font-size: 9.5pt;"
)

#: The little labels over a row of switches.
ROW_LABEL_STYLE = (
    f"color: {theme.TEXT_DIM}; font-size: 9pt; font-weight: 600;"
)

#: Where a derived number says where it came from.
DERIVED_STYLE = f"color: {theme.AQUA}; font-size: 9pt;"

#: ...and where it says there was nothing to derive it from.
UNDERIVED_STYLE = f"color: {theme.AMBER}; font-size: 9pt;"

#: Eight jobs as two rows of four.
CARDS_PER_ROW = 4

#: Settings per row, matching the three-column shape of the source app.
COLUMNS = 3

#: Editors may shrink with the pane, but not below something clickable.
MIN_EDITOR_WIDTH = 150

#: A spin box needs an upper bound; nothing here is a count that large.
BIG = 10_000_000


# ----------------------------------------------------------------
# The boxes at the top
# ----------------------------------------------------------------


class CardButton(QFrame):
    """
    One clickable box: a title, optionally a line saying what it is for,
    and a border that lights up when it is the one selected.

    A push button would do the job with a lot less code, but not with two
    lines of type at two weights -- and which job to run is the first
    decision anyone makes here, so it is worth the room. An empty hint
    leaves the second line out altogether rather than reserving blank
    space for it, which is how the INCAR dialog keeps its cards to the
    job name alone.
    """

    #: The card's own title, so one handler can serve all of them.
    picked = Signal(str)

    def __init__(self, title: str, hint: str = "", parent=None) -> None:

        super().__init__(parent)

        self.title = title

        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("on", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(2)

        name = QLabel(title.replace("&", "&&"))
        name.setObjectName("cardTitle")

        layout.addWidget(name)

        if hint:

            caption = QLabel(hint)
            caption.setObjectName("cardHint")
            caption.setWordWrap(True)

            layout.addWidget(caption)

        layout.addStretch(1)

    def set_checked(self, on: bool) -> None:
        """Light the border, or put it out."""

        self.setProperty("on", on)

        # A dynamic property in a stylesheet is only re-read when the
        # widget is re-polished; without this the border never changes.
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt's name

        if event.button() == Qt.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.picked.emit(self.title)

        super().mouseReleaseEvent(event)


def _switch(text: str, tooltip: str) -> QPushButton:
    """One toggle in a row of them."""

    button = QPushButton(text.replace("&", "&&"))

    button.setObjectName("chip")
    button.setCheckable(True)
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(tooltip)
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    return button


def _row_label(text: str) -> QLabel:

    label = QLabel(text)
    label.setStyleSheet(ROW_LABEL_STYLE)

    return label


def _heading(text: str) -> QLabel:

    label = QLabel(text)
    label.setStyleSheet(SECTION_STYLE)

    return label


# ----------------------------------------------------------------


class IncarDialog(QDialog):
    """Assemble an INCAR for one of the eight main jobs."""

    def __init__(
        self,
        elements: list[str] | None = None,
        counts: list[int] | None = None,
        parent=None,
        job: str | None = None,
    ) -> None:

        super().__init__(parent)

        #: Species in POSCAR order. The per-species tags are written
        #: against this order and are left out entirely without it, since
        #: a short list is read positionally and applied to the wrong
        #: element rather than rejected.
        self.elements = tuple(elements or [])

        #: How many ions of each, in the same order. Only the band-count
        #: estimate needs it.
        self.counts = tuple(counts or [])

        chosen = ic.job_by_name(job or "") or ic.JOBS[0]

        self.job = chosen.name

        #: Every setting of every job, so that clicking away from a job
        #: and back again does not throw away what was typed into it.
        #: The keys are prefixed per job, so they cannot collide.
        self.values: dict[str, str] = {}

        for definition in ic.JOBS:
            self.values.update(definition.defaults())

        self.values.update(ic.global_defaults())
        self.values.update(ic.module_defaults())

        #: Which single-point boxes are ticked. One to start with,
        #: because an empty selection is not a calculation.
        self.subjobs: set[str] = {"Static SCF"}

        #: Which optional modules are on.
        self.modules: set[str] = set()

        #: Which species carry a Hubbard U. Nothing until +U is switched
        #: on and a species is picked -- leaving one out is a choice, and
        #: it should be a deliberate one.
        self.hubbard: set[str] = set()

        #: True while the pane is being redrawn. Tearing down a line edit
        #: takes its focus away, which emits ``editingFinished``, which
        #: would start another redraw from inside this one.
        self._building = False

        self._seed_per_species()
        self._seed_bands()

        self.cards: dict[str, CardButton] = {}
        self.subjob_buttons: dict[str, QPushButton] = {}
        self.module_buttons: dict[str, QPushButton] = {}
        self.hubbard_buttons: dict[str, QPushButton] = {}

        #: Field name to the widget editing it.
        self.editors: dict[str, QWidget] = {}

        self.setWindowTitle("INCAR")
        fit_to_screen(self, 1040, 700)
        self.resize(1320, 900)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(self._build_cards())
        layout.addWidget(_hline())
        layout.addWidget(self._build_structure())
        layout.addWidget(_hline())

        splitter = QSplitter(Qt.Horizontal)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([720, 600])

        layout.addWidget(splitter, 1)

        layout.addLayout(self._build_buttons())

        self._select(self.job)

    # ------------------------------------------------------------
    # Starting values that depend on the structure
    # ------------------------------------------------------------

    def _seed_per_species(self) -> None:
        """
        One starting value per species for the per-species tags.

        Done once, up front, so that a species added later is impossible
        -- the structure is fixed for the life of the dialog.
        """

        for definition in ic.JOBS:

            for field in definition.fields():

                if not field.per_species:
                    continue

                for element in self.elements:
                    self.values[f"{field.name}_{element}"] = field.default

        for element in self.elements:

            u = ic.TRANSITION_METAL_U.get(element)

            if u is not None:
                self.values[f"dftu_u_{element}"] = f"{u:.1f}"

    def _seed_bands(self) -> None:
        """
        Start the optical band count at the estimate from the structure.

        VASP's own default is the electron count plus a small margin,
        which is thin for a spectrum: the transitions that matter go into
        empty states well above the gap, and a band count that stops just
        above the Fermi level cuts the spectrum off without saying so.
        """

        bands = self._setup().nbands()

        if bands is not None:
            self.values["optical_nbands"] = str(bands)

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_cards(self) -> QWidget:
        """The eight boxes."""

        host = QWidget()

        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)

        for index, definition in enumerate(ic.JOBS):

            # Title only. What the job is for is on hover, so eight cards
            # read as eight names rather than a paragraph.
            card = CardButton(definition.name)
            card.setToolTip(definition.tagline)
            card.picked.connect(self._select)

            grid.addWidget(
                card, index // CARDS_PER_ROW, index % CARDS_PER_ROW
            )

            self.cards[definition.name] = card

        for column in range(CARDS_PER_ROW):
            grid.setColumnStretch(column, 1)

        return host

    def _build_structure(self) -> QWidget:
        """
        What is open, and in what order.

        The order is the part worth showing. LDAUL, LDAUU, LDAUJ and
        LANGEVIN_GAMMA are all read positionally against the POTCAR, so
        the species order is what makes those lines mean what they say.
        """

        host = QWidget()

        row = QVBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        row.addWidget(_row_label("STRUCTURE"))

        if self.elements:

            counts = dict(zip(self.elements, self.counts))

            body = QLabel(
                "   ".join(
                    f"{element} x{counts.get(element, 0)}"
                    for element in self.elements
                )
                + "          "
                + " → ".join(self.elements)
            )
            body.setStyleSheet(DERIVED_STYLE)

        else:

            body = QLabel("None open")
            body.setStyleSheet(UNDERIVED_STYLE)

        body.setToolTip(
            "The per-species tags (LDAUL, LDAUU, LDAUJ, LANGEVIN_GAMMA) "
            "are read positionally against the POTCAR, so this order is "
            "what makes those lines mean what they say. Without a "
            "structure they are left out rather than guessed at."
        )

        row.addWidget(body)

        return host

    def _build_left(self) -> QWidget:
        """The switches and the settings, scrolling together."""

        area = QScrollArea()
        area.setWidgetResizable(True)

        host = QWidget()

        self.left_layout = QVBoxLayout(host)
        self.left_layout.setContentsMargins(2, 2, 10, 2)
        self.left_layout.setSpacing(10)

        # Replaced wholesale on every change that alters what is on show.
        self.body = QWidget()

        self.left_layout.addWidget(self.body)
        self.left_layout.addStretch(1)

        area.setWidget(host)

        return area

    def _build_right(self) -> QWidget:
        """The file itself, and nothing beside it."""

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)

        column.addWidget(QLabel("INCAR"))

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setStyleSheet(MONO_STYLE)
        self.preview.setLineWrapMode(QPlainTextEdit.NoWrap)

        column.addWidget(self.preview, 1)

        return host

    def _build_buttons(self) -> QHBoxLayout:

        buttons = QHBoxLayout()

        reset = QPushButton("Reset")
        reset.setStyleSheet(INFO_BUTTON_STYLE)
        reset.setToolTip(
            "Back to this job as it comes: no additions, no edits."
        )
        reset.clicked.connect(self._reset)

        buttons.addWidget(reset)
        buttons.addStretch(1)

        copy = QPushButton("Copy")
        copy.clicked.connect(self._copy)

        buttons.addWidget(copy)

        save = QPushButton("Save INCAR...")
        save.setStyleSheet(SUCCESS_BUTTON_STYLE)
        save.clicked.connect(self._save)

        buttons.addWidget(save)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        buttons.addWidget(close)

        return buttons

    # ------------------------------------------------------------
    # Choosing a job
    # ------------------------------------------------------------

    def _select(self, name: str) -> None:
        """A card was clicked, or the dialog has just opened."""

        if ic.job_by_name(name) is None:
            return

        self.job = name

        for title, card in self.cards.items():
            card.set_checked(title == name)

        self._rebuild()

    def _reset(self) -> None:
        """This job back to its defaults, leaving the others alone."""

        definition = ic.job_by_name(self.job)

        if definition is None:
            return

        self.values.update(definition.defaults())
        self.values.update(ic.global_defaults())
        self.values.update(ic.module_defaults())

        self._seed_per_species()
        self._seed_bands()

        self.modules.clear()
        self.hubbard.clear()
        self.subjobs = {"Static SCF"}

        self._rebuild()

    # ------------------------------------------------------------
    # Laying the settings out
    # ------------------------------------------------------------

    def _rebuild(self) -> None:
        """
        Redraw everything under the cards.

        Wholesale rather than in place, because what is on show depends on
        what has been chosen -- the thermostat decides whether SMASS
        exists at all -- and reconciling that incrementally is more code
        than it is worth for a pane this size.
        """

        definition = ic.job_by_name(self.job)

        if definition is None or self._building:
            return

        self._building = True

        self.editors.clear()
        self.subjob_buttons.clear()
        self.module_buttons.clear()
        self.hubbard_buttons.clear()

        fresh = QWidget()

        column = QVBoxLayout(fresh)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)

        if definition.name == "Single-Point":
            column.addWidget(self._build_subjobs())
            column.addWidget(_hline())

        for section in self._sections(definition):
            column.addWidget(self._build_section(section))

        if definition.name == "Single-Point":

            bands = self._build_bands()

            if bands is not None:
                column.addWidget(bands)

        column.addWidget(_hline())
        column.addWidget(self._build_modules())

        for name in ic.MODULES:

            if name not in self.modules:
                continue

            if name == "DFT+U":

                hubbard = self._build_hubbard()

                if hubbard is not None:
                    column.addWidget(hubbard)

                continue

            section = ic.MODULE_SECTIONS.get(name)

            if section is not None:
                column.addWidget(self._build_section(section))

        column.addWidget(_hline())

        column.addWidget(_heading("Global VASP Parameters"))

        for section in ic.GLOBAL_SECTIONS:
            column.addWidget(self._build_section(section))

        self.left_layout.replaceWidget(self.body, fresh)
        self.body.deleteLater()
        self.body = fresh

        self._building = False

        self._refresh()

    def _sections(self, definition) -> list:
        """
        The job's sections that currently apply.

        A single point filters by which boxes are ticked; every other job
        filters on the value of whatever field the section names.
        """

        if definition.name == "Single-Point":

            wanted = ic.single_point_sections(self._ordered_subjobs())

            return [
                section
                for section in definition.sections
                if section.title in wanted
            ]

        return [
            section
            for section in definition.sections
            if section.visible(self.values)
        ]

    def _build_section(self, section) -> QWidget:
        """One titled run of settings, three to a row."""

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        column.addWidget(_heading(section.title))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        index = 0

        for field in section.fields:

            for cell in self._cells(field):

                grid.addWidget(cell, index // COLUMNS, index % COLUMNS)
                index += 1

        for spare in range(COLUMNS):
            grid.setColumnStretch(spare, 1)

        column.addLayout(grid)

        return host

    def _cells(self, field) -> list[QWidget]:
        """
        The editors one field needs.

        Usually one. A per-species tag needs one per element, and none at
        all when there is no structure open -- the tag is then left out
        of the file rather than written against a guessed species order.
        """

        if not field.per_species:
            return [self._cell(field.name, field.label, field)]

        return [
            self._cell(
                f"{field.name}_{element}",
                f"{field.label} - {element}",
                field,
            )
            for element in self.elements
        ]

    def _cell(self, name: str, label: str, field) -> QWidget:
        """
        The tag's name and its editor, and nothing else on screen.

        What the tag does is on the editor as a tooltip rather than
        printed under it. A caption under every one of two hundred
        settings is a wall of prose to read past on the way to the one
        being changed, and the pane is easier to scan without them.
        """

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        title = QLabel(label)
        title.setStyleSheet(MONO_STYLE + " font-weight: bold;")

        column.addWidget(title)

        editor = self._editor(name, field)
        editor.setMinimumWidth(MIN_EDITOR_WIDTH)

        if field.note:
            editor.setToolTip(field.note)
            title.setToolTip(field.note)

        self.editors[name] = editor

        column.addWidget(editor)

        column.addStretch(1)

        return host

    def _editor(self, name: str, field) -> QWidget:
        """The widget for one setting, wired to update the file."""

        value = self.values.get(name, field.default)

        if field.kind is ic.Kind.FIXED:

            box = QLineEdit(value)
            box.setReadOnly(True)
            box.setEnabled(False)
            box.setStyleSheet(MONO_STYLE)

            return box

        if field.kind is ic.Kind.CHOICE:

            box = QComboBox()

            for choice in field.choices:
                box.addItem(choice.label, choice.value)

            found = box.findData(value)
            box.setCurrentIndex(found if found >= 0 else 0)

            box.currentIndexChanged.connect(
                lambda _, key=name, widget=box: self._changed(
                    key, widget
                )
            )

            return box

        if field.kind is ic.Kind.BOOL:

            box = QCheckBox()
            box.setChecked(value == ic.TRUE)
            box.setText("on" if value == ic.TRUE else "off")

            box.toggled.connect(
                lambda on, key=name, widget=box: (
                    widget.setText("on" if on else "off"),
                    self._changed(key, widget),
                )
            )

            return box

        if field.kind is ic.Kind.INT:

            box = QSpinBox()
            box.setRange(
                int(field.low) if field.low is not None else -BIG,
                int(field.high) if field.high is not None else BIG,
            )
            box.setSingleStep(max(1, int(field.step)))
            box.setValue(_as_int(value, _as_int(field.default, 0)))

            box.valueChanged.connect(
                lambda _, key=name, widget=box: self._changed(key, widget)
            )

            return box

        if field.kind is ic.Kind.FLOAT:

            box = QDoubleSpinBox()
            box.setDecimals(field.decimals)
            box.setRange(
                field.low if field.low is not None else -float(BIG),
                field.high if field.high is not None else float(BIG),
            )
            box.setSingleStep(field.step)
            box.setValue(_as_float(value, _as_float(field.default, 0.0)))

            box.valueChanged.connect(
                lambda _, key=name, widget=box: self._changed(key, widget)
            )

            return box

        box = QLineEdit(value)

        # On editingFinished rather than textChanged: EDIFF is typed as
        # ``1E-6`` and passes through ``1``, ``1E`` and ``1E-`` on the
        # way, and rewriting the file at each of those is noise.
        box.editingFinished.connect(
            lambda key=name, widget=box: self._changed(key, widget)
        )

        return box

    # ------------------------------------------------------------
    # Switch rows
    # ------------------------------------------------------------

    def _build_subjobs(self) -> QWidget:
        """The single-point boxes. Any number of them at once."""

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        column.addWidget(_heading("Single-Point Calculations"))

        row, flow = flow_widget()

        for name in ic.SUBJOBS:

            button = _switch(name, ic.SUBJOB_NOTES.get(name, ""))
            button.setChecked(name in self.subjobs)
            button.toggled.connect(
                lambda on, title=name: self._toggle(self.subjobs, title, on)
            )

            flow.addWidget(button)

            self.subjob_buttons[name] = button

        column.addWidget(row)

        return host

    def _build_modules(self) -> QWidget:
        """The optional physics. None of it on until asked for."""

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        column.addWidget(_heading("Optional Tags"))

        row, flow = flow_widget()

        for name in ic.MODULES:

            button = _switch(name, ic.MODULE_NOTES.get(name, ""))
            button.setChecked(name in self.modules)
            button.toggled.connect(
                lambda on, title=name: self._toggle(self.modules, title, on)
            )

            flow.addWidget(button)

            self.module_buttons[name] = button

        column.addWidget(row)

        return host

    def _build_hubbard(self) -> QWidget | None:
        """
        Which species carry a U, and how much.

        Only the transition metals actually present are offered, and the
        three lists are always written for every species in POSCAR order
        -- including the ones with no correction, which get ``-1``. A
        list written only for the corrected species is read positionally
        all the same, and applies the U to whatever happens to sit in
        that slot.
        """

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        column.addWidget(_heading("DFT+U Element Selection"))

        if not self.elements:

            note = QLabel("No structure open")
            note.setStyleSheet(UNDERIVED_STYLE)
            note.setToolTip(
                "The +U lists are read positionally against the POTCAR, "
                "so without a species order they are left out rather "
                "than guessed at."
            )

            column.addWidget(note)

            return host

        detected = ic.transition_metals(self.elements)

        if not detected:

            note = QLabel("No transition metals present")
            note.setStyleSheet(UNDERIVED_STYLE)

            column.addWidget(note)

            return host

        row, flow = flow_widget()

        for element in detected:

            button = _switch(
                element,
                f"Tabulated U for {element}: "
                f"{ic.TRANSITION_METAL_U[element]:.1f} eV",
            )
            button.setChecked(element in self.hubbard)
            button.toggled.connect(
                lambda on, symbol=element: self._toggle(
                    self.hubbard, symbol, on
                )
            )

            flow.addWidget(button)

            self.hubbard_buttons[element] = button

        column.addWidget(row)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        index = 0

        for element in detected:

            if element not in self.hubbard:
                continue

            field = ic.Field(
                f"dftu_u_{element}",
                f"LDAUU - {element}",
                ic.Kind.FLOAT,
                f"{ic.TRANSITION_METAL_U[element]:.1f}",
                low=0.0,
                step=0.1,
                decimals=1,
                note=(
                    f"Tabulated: {ic.TRANSITION_METAL_U[element]:.1f} eV. "
                    "Dudarev, so this is U - J."
                ),
            )

            grid.addWidget(
                self._cell(field.name, field.label, field),
                index // COLUMNS,
                index % COLUMNS,
            )

            index += 1

        for spare in range(COLUMNS):
            grid.setColumnStretch(spare, 1)

        column.addLayout(grid)

        return host

    def _build_bands(self) -> QWidget | None:
        """
        The band count, and the sum behind it.

        Shown rather than editable because it is an estimate with a
        stated basis: so many bands per atom by block. The breakdown is
        underneath so that an element the classification does not cover
        shows up as a zero instead of quietly shrinking the total.
        """

        setup = self._setup()

        doubled = "COHP / ICOHP" in self.subjobs

        bands = setup.nbands(doubled=doubled)

        host = QWidget()

        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        column.addWidget(_heading("Automatic NBANDS"))

        if bands is None:

            note = QLabel("No structure open")
            note.setStyleSheet(UNDERIVED_STYLE)
            note.setToolTip(
                "NBANDS is left out, so VASP picks its own -- which is "
                "thin for anything reading empty states."
            )

            column.addWidget(note)

            return host

        lines = [
            f"{element}: {count} x {each} ({block}) = {total}"
            for element, count, each, block, total in setup.band_breakdown()
        ]

        plain = setup.nbands()

        if doubled and plain is not None:
            lines.append(f"COHP needs empty states: 2 x {plain} = {bands}")

        box = QLineEdit(str(bands))
        box.setReadOnly(True)
        box.setEnabled(False)
        box.setStyleSheet(MONO_STYLE)
        box.setMaximumWidth(200)

        # The sum behind the number, on hover. It is worth being able to
        # check -- an element the block classification does not cover
        # contributes zero and would otherwise shrink the total silently
        # -- but it is a derivation, not something to read every time.
        box.setToolTip("\n".join(lines))

        column.addWidget(box)

        return host

    # ------------------------------------------------------------
    # Reacting
    # ------------------------------------------------------------

    def _toggle(self, group: set, name: str, on: bool) -> None:
        """A chip was clicked."""

        if on:
            group.add(name)

        else:
            group.discard(name)

        # Every chip here reveals or hides something: a sub-job brings its
        # own settings, a module its own block, a +U species its own
        # editor.
        self._rebuild()

    def _changed(self, name: str, editor: QWidget) -> None:
        """
        An editor moved.

        The widget is passed in rather than looked up by name. Hiding a
        section tears its editors down, and a widget on its way out can
        still emit -- ``editingFinished`` fires when a line edit loses
        focus, which is exactly what being removed does to it. Reading
        the value back out of ``self.editors`` at that moment finds
        whatever replaced it and writes that value under this name.
        """

        if self._building or self.editors.get(name) is not editor:
            return

        self.values[name] = _read(editor)

        if name == "hse_type":
            self._apply_hybrid()

        if name in self._gates():
            self._rebuild()
            return

        self._refresh()

    def _gates(self) -> set[str]:
        """
        Fields whose value decides what else is on show.

        Anything a section names in its condition, plus the hybrid
        dropdown, which rewrites the two numbers under it.
        """

        definition = ic.job_by_name(self.job)

        names = {"hse_type"}

        if definition is not None:

            for section in definition.sections:
                names.update(section.gates())

        return names

    def _apply_hybrid(self) -> None:
        """
        Put the chosen hybrid's numbers into AEXX and HFSCREEN.

        PBE0 and HSE06 differ in exactly one of them -- the screening --
        and leaving that at 0.2 after picking PBE0 gives HSE06 under
        another name. Custom leaves both alone, which is the point of it.
        """

        preset = ic.HYBRID_PRESETS.get(self.values.get("hse_type", ""))

        if preset is None:
            return

        self.values["hse_aexx"], self.values["hse_hfscreen"] = preset

    def _ordered_subjobs(self) -> tuple[str, ...]:
        """
        The ticked boxes, in the order they are offered.

        A set has no order and the file says which sub-jobs it was for,
        so it is put back into a fixed one; otherwise the comment line
        shuffles between runs and every INCAR looks changed.
        """

        return tuple(name for name in ic.SUBJOBS if name in self.subjobs)

    def _setup(self) -> ic.Setup:
        """Everything the service module needs to build the file."""

        return ic.Setup(
            job=self.job,
            values=self.values,
            subjobs=self._ordered_subjobs(),
            modules=tuple(
                name for name in ic.MODULES if name in self.modules
            ),
            elements=self.elements,
            counts=self.counts,
            hubbard=tuple(
                element
                for element in self.elements
                if element in self.hubbard
            ),
        )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    def text(self) -> str:
        """The INCAR as it stands."""

        return ic.incar_text(self._setup())

    def _refresh(self) -> None:

        self.preview.setPlainText(self.text())

    def _copy(self) -> None:
        """Put the file on the clipboard, for pasting into a terminal."""

        QGuiApplication.clipboard().setText(self.text())

    def _save(self) -> None:

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save INCAR",
            "INCAR",
            "INCAR File (INCAR);;All Files (*)",
        )

        if not filename:
            return

        try:
            Path(filename).write_text(self.text(), encoding="utf-8")

        except OSError as exc:
            QMessageBox.critical(
                self, "Save failed", f"Could not write {filename}: {exc}"
            )
            return

        QMessageBox.information(
            self, "Saved", f"{self.job} INCAR written to\n{filename}"
        )


# ----------------------------------------------------------------
# Reading widgets back
# ----------------------------------------------------------------


def _read(editor: QWidget) -> str:
    """The value an editor holds, as it will be written."""

    if isinstance(editor, QComboBox):
        return str(editor.currentData())

    if isinstance(editor, QCheckBox):
        return ic.TRUE if editor.isChecked() else ic.FALSE

    if isinstance(editor, QDoubleSpinBox):
        return f"{editor.value():.{editor.decimals()}f}"

    if isinstance(editor, QSpinBox):
        return str(editor.value())

    if isinstance(editor, QLineEdit):
        return editor.text()

    return ""


def _as_int(value: str, fallback: int) -> int:

    try:
        return int(float(value))

    except ValueError:
        return fallback


def _as_float(value: str, fallback: float) -> float:

    try:
        return float(value)

    except ValueError:
        return fallback
