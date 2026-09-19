"""
Which element pairs are drawn as bonded.

One row per pair of elements in the open structure, each with a switch
and a distance. The defaults come from the covalent radii, which is the
right answer often enough to be the default and wrong often enough to
need this window -- see :mod:`qvista.visualization.bonding` for the case
that forces it, an intercalated cathode whose alkali metal is drawn
bonded to the framework it merely sits inside.

Changes apply to the viewport as they are made rather than on OK. The
whole judgement being made here is whether the picture reads correctly,
and that cannot be answered from a dialog that hides the picture until
it is dismissed. Closing keeps them; Reset puts every pair back on its
radii.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
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
from qvista.visualization.bonding import (
    MAX_CUTOFF,
    MIN_CUTOFF,
    BondRules,
    element_pairs,
)

#: The pair name, so the column reads as a column.
PAIR_STYLE = theme.MONO_STYLE + " font-weight: bold;"

#: Column headings.
HEAD_STYLE = (
    f"color: {theme.TEXT_DIM}; font-size: 9pt; font-weight: 600;"
)

#: Where a row says it is no longer on its default.
CHANGED_STYLE = f"color: {theme.AQUA}; font-size: 9pt;"

#: ...and where it says it is.
DEFAULT_STYLE = f"color: {theme.TEXT_DIM}; font-size: 9pt;"

#: Enough for a distance in Angstroms and no more.
CUTOFF_WIDTH = 110


class BondDialog(QDialog):
    """Per-element-pair bonding rules."""

    #: The edited rules, emitted on every change so the viewport can
    #: follow along.
    rulesChanged = Signal(BondRules)

    def __init__(
        self,
        elements,
        rules: BondRules | None = None,
        parent=None,
    ) -> None:

        super().__init__(parent)

        #: Species in the open structure. The pairs are built from these,
        #: so a rule can only ever be made for something on screen.
        self.elements = tuple(elements or [])

        #: Edited in place and emitted. A copy, so that the caller's
        #: rules are only replaced by what it does with the signal.
        self.rules = (rules or BondRules()).copy()

        self.switches: dict[tuple[str, str], QCheckBox] = {}
        self.cutoffs: dict[tuple[str, str], QDoubleSpinBox] = {}
        self.states: dict[tuple[str, str], QLabel] = {}

        #: True while the rows are being written from the rules, so that
        #: setting a widget's value does not read straight back out of it
        #: as though the user had done it.
        self._loading = False

        self.setWindowTitle("Bonds")
        fit_to_screen(self, 520, 360)
        self.resize(620, 560)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(self._build_head())
        layout.addWidget(_hline())
        layout.addWidget(self._build_rows(), 1)
        layout.addLayout(self._build_buttons())

        self._load()

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def _build_head(self) -> QWidget:

        host = QWidget()

        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)

        if self.elements:
            title = QLabel("   ".join(self.elements))

        else:
            title = QLabel("No structure open")

        title.setStyleSheet(PAIR_STYLE)

        row.addWidget(title)
        row.addStretch(1)

        return host

    def _build_rows(self) -> QWidget:
        """One row per pair, inside a scroll area."""

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        host = QWidget()

        grid = QGridLayout(host)
        grid.setContentsMargins(2, 2, 8, 2)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        for column, name in enumerate(("Pair", "Show", "Cutoff (Å)", "")):

            head = QLabel(name)
            head.setStyleSheet(HEAD_STYLE)

            grid.addWidget(head, 0, column)

        pairs = element_pairs(self.elements)

        for index, pair in enumerate(pairs, start=1):

            a, b = pair

            name = QLabel(f"{a} – {b}")
            name.setStyleSheet(PAIR_STYLE)

            grid.addWidget(name, index, 0)

            switch = QCheckBox()
            switch.toggled.connect(
                lambda on, key=pair: self._toggled(key, on)
            )

            grid.addWidget(switch, index, 1)

            cutoff = QDoubleSpinBox()
            cutoff.setRange(MIN_CUTOFF, MAX_CUTOFF)
            cutoff.setDecimals(2)
            cutoff.setSingleStep(0.05)
            cutoff.setSuffix(" Å")
            cutoff.setFixedWidth(CUTOFF_WIDTH)
            cutoff.valueChanged.connect(
                lambda value, key=pair: self._cutoff_changed(key, value)
            )

            grid.addWidget(cutoff, index, 2)

            state = QLabel()

            grid.addWidget(state, index, 3)

            self.switches[pair] = switch
            self.cutoffs[pair] = cutoff
            self.states[pair] = state

        grid.setColumnStretch(3, 1)
        grid.setRowStretch(len(pairs) + 1, 1)

        area.setWidget(host)

        return area

    def _build_buttons(self) -> QHBoxLayout:

        buttons = QHBoxLayout()

        reset = QPushButton("Reset")
        reset.setStyleSheet(INFO_BUTTON_STYLE)
        reset.setToolTip("Every pair back on its covalent radii.")
        reset.clicked.connect(self._reset)

        buttons.addWidget(reset)

        none = QPushButton("Hide All")
        none.setToolTip("Switch every pair off.")
        none.clicked.connect(lambda: self._set_all(False))

        buttons.addWidget(none)

        every = QPushButton("Show All")
        every.setToolTip("Switch every pair on, keeping the cutoffs.")
        every.clicked.connect(lambda: self._set_all(True))

        buttons.addWidget(every)

        buttons.addStretch(1)

        close = QPushButton("Close")
        close.setStyleSheet(SUCCESS_BUTTON_STYLE)
        close.clicked.connect(self.accept)

        buttons.addWidget(close)

        return buttons

    # ------------------------------------------------------------
    # Rules to widgets
    # ------------------------------------------------------------

    def _load(self) -> None:
        """Write every row from the rules."""

        self._loading = True

        for pair, switch in self.switches.items():

            a, b = pair

            switch.setChecked(self.rules.is_enabled(a, b))

            self.cutoffs[pair].setValue(self.rules.cutoff(a, b))

            self._refresh_state(pair)

        self._loading = False

    def _refresh_state(self, pair: tuple[str, str]) -> None:
        """
        Say whether the row is still on its default, and what that was.

        Worth showing because a cutoff dragged to 2.40 A means nothing on
        its own -- what matters is whether that is wider or narrower than
        the radii would have given, and by how much.
        """

        a, b = pair

        label = self.states[pair]

        default = self.rules.default_cutoff(a, b)

        if self.rules.is_default(a, b):

            label.setText(f"default {default:.2f} Å")
            label.setStyleSheet(DEFAULT_STYLE)

            return

        if not self.rules.is_enabled(a, b):

            label.setText("hidden")
            label.setStyleSheet(CHANGED_STYLE)

            return

        label.setText(f"default was {default:.2f} Å")
        label.setStyleSheet(CHANGED_STYLE)

    # ------------------------------------------------------------
    # Widgets to rules
    # ------------------------------------------------------------

    def _toggled(self, pair: tuple[str, str], on: bool) -> None:

        if self._loading:
            return

        self.rules.set_enabled(pair[0], pair[1], on)

        self._refresh_state(pair)

        self._emit()

    def _cutoff_changed(
        self, pair: tuple[str, str], value: float
    ) -> None:

        if self._loading:
            return

        self.rules.set_cutoff(pair[0], pair[1], value)

        self._refresh_state(pair)

        self._emit()

    def _reset(self) -> None:
        """Every pair back on its radii."""

        self.rules.reset()

        self._load()
        self._emit()

    def _set_all(self, on: bool) -> None:
        """
        Switch every pair on or off at once.

        Hiding all of them is how a picture of an intercalation cathode
        usually starts: switch the lot off, then bring back the one
        framework bond that carries the structure.
        """

        for a, b in element_pairs(self.elements):
            self.rules.set_enabled(a, b, on)

        self._load()
        self._emit()

    def _emit(self) -> None:

        self.rulesChanged.emit(self.rules)
