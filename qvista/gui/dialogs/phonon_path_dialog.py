"""
Choose the path a phonon dispersion is drawn along.

Deliberately not the Crystallography path builder. That one describes a
structure, draws the Brillouin zone and writes KPOINTS; this one changes
a figure whose force constants already exist, so the only question is
which line through the zone to evaluate along. A zone plot here would be
a second thing to look at while the figure it is about sits behind the
dialog.

The path is built by clicking labels in order. Consecutive clicks make a
continuous stretch; "Break" starts a new one, which is how a path that
jumps -- X to K without a line between them -- is expressed, and it
draws as the ``X|K`` tick the axis already uses.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
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
from qvista.services.phonon_path import (
    DEFAULT_NPOINTS,
    MAX_NPOINTS,
    MIN_NPOINTS,
    PathOptions,
    path_label,
)

#: The path as it currently reads.
PATH_STYLE = theme.MONO_STYLE + f" color: {theme.AQUA}; font-weight: 600;"

#: Section headings.
HEAD_STYLE = (
    f"color: {theme.TEXT_DIM}; font-size: 9pt; font-weight: 600;"
)

#: Where a break is shown in the built path.
BREAK = "|"


class PhononPathDialog(QDialog):
    """Build a q-path from this lattice's high-symmetry points."""

    #: The chosen path, as consecutive ``(start, end)`` label pairs,
    #: and the q-points per segment.
    path_chosen = Signal(list, int)

    def __init__(
        self,
        options: PathOptions,
        current=None,
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.options = options

        #: Labels in the order they were clicked. ``BREAK`` marks where
        #: the path jumps rather than continuing.
        self.sequence: list[str] = []

        self.setWindowTitle("Band Path")
        fit_to_screen(self, 460, 380)
        self.resize(560, 480)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(self._build_body(), 1)

        layout.addWidget(_hline())

        layout.addWidget(_heading("Path"))

        self.readout = QLabel()
        self.readout.setStyleSheet(PATH_STYLE)
        self.readout.setWordWrap(True)
        self.readout.setMinimumHeight(38)

        layout.addWidget(self.readout)

        layout.addLayout(self._build_buttons())

        self._load(current or options.recommended)

    # ------------------------------------------------------------

    def _build_body(self) -> QWidget:

        host = QWidget()

        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        # --- available points ---

        left = QVBoxLayout()

        left.addWidget(_heading("High-symmetry points"))

        self.available = QListWidget()

        for name in self.options.labels():

            item = QListWidgetItem(name)
            item.setToolTip(
                "  ".join(
                    f"{value:+.4f}" for value in self.options.points[name]
                )
            )

            self.available.addItem(item)

        self.available.itemDoubleClicked.connect(
            lambda item: self._append(item.text())
        )

        left.addWidget(self.available, 1)

        add = QPushButton("Add ->")
        add.setToolTip("Add the selected point to the end of the path.")
        add.clicked.connect(self._add_selected)

        left.addWidget(add)

        row.addLayout(left, 1)

        # --- the path being built ---

        right = QVBoxLayout()

        right.addWidget(_heading("In order"))

        self.chosen = QListWidget()

        right.addWidget(self.chosen, 1)

        buttons = QHBoxLayout()

        for text, tip, slot in (
            (
                "Break",
                "Start a new stretch. The path jumps here rather than "
                "drawing a line, which is the X|K tick on the axis.",
                self._add_break,
            ),
            ("Remove", "Drop the last entry.", self._remove_last),
            ("Clear", "Start again.", self._clear),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)

            buttons.addWidget(button)

        right.addLayout(buttons)

        row.addLayout(right, 1)

        return host

    def _build_buttons(self) -> QHBoxLayout:

        buttons = QHBoxLayout()

        recommended = QPushButton("Recommended")
        recommended.setStyleSheet(INFO_BUTTON_STYLE)
        recommended.setToolTip("Back to seekpath's path for this lattice.")
        recommended.clicked.connect(
            lambda: self._load(self.options.recommended)
        )

        buttons.addWidget(recommended)

        buttons.addSpacing(12)

        buttons.addWidget(QLabel("Points per segment"))

        self.npoints = QSpinBox()
        self.npoints.setRange(MIN_NPOINTS, MAX_NPOINTS)
        self.npoints.setValue(DEFAULT_NPOINTS)
        self.npoints.setToolTip(
            "How finely each segment is sampled. Higher is smoother and "
            "costs a diagonalisation per point."
        )

        buttons.addWidget(self.npoints)

        buttons.addStretch(1)

        self.plot_button = QPushButton("Plot")
        self.plot_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.plot_button.clicked.connect(self._accept)

        buttons.addWidget(self.plot_button)

        close = QPushButton("Cancel")
        close.clicked.connect(self.reject)

        buttons.addWidget(close)

        return buttons

    # ------------------------------------------------------------
    # Building
    # ------------------------------------------------------------

    def _load(self, segments) -> None:
        """Show an existing path as a sequence of clicks."""

        self.sequence = []

        previous = None

        for start, end in segments:

            if previous is None:
                self.sequence.append(start)

            elif start != previous:
                self.sequence.append(BREAK)
                self.sequence.append(start)

            self.sequence.append(end)

            previous = end

        self._refresh()

    def _add_selected(self) -> None:

        item = self.available.currentItem()

        if item is not None:
            self._append(item.text())

    def _append(self, name: str) -> None:

        self.sequence.append(name)

        self._refresh()

    def _add_break(self) -> None:

        # Nothing to break away from, and two in a row means nothing.
        if not self.sequence or self.sequence[-1] == BREAK:
            return

        self.sequence.append(BREAK)

        self._refresh()

    def _remove_last(self) -> None:

        if self.sequence:
            self.sequence.pop()

        self._refresh()

    def _clear(self) -> None:

        self.sequence = []

        self._refresh()

    # ------------------------------------------------------------

    def segments(self) -> list[tuple[str, str]]:
        """
        The built sequence as consecutive label pairs.

        A break ends one stretch and starts the next, so no pair ever
        spans it -- which is what stops a jump being drawn as a line.
        """

        pairs: list[tuple[str, str]] = []

        previous = None

        for name in self.sequence:

            if name == BREAK:
                previous = None
                continue

            if previous is not None:
                pairs.append((previous, name))

            previous = name

        return pairs

    def _refresh(self) -> None:

        self.chosen.clear()

        for name in self.sequence:
            self.chosen.addItem(name)

        pairs = self.segments()

        self.readout.setText(path_label(pairs) if pairs else "(empty)")

        # A single point is not a path, and neither is nothing.
        self.plot_button.setEnabled(bool(pairs))

    def _accept(self) -> None:

        pairs = self.segments()

        if not pairs:
            return

        self.path_chosen.emit(pairs, self.npoints.value())

        self.accept()


def _heading(text: str) -> QLabel:

    label = QLabel(text)
    label.setStyleSheet(HEAD_STYLE)

    return label
