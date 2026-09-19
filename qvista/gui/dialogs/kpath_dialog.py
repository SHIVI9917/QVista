"""
Dialogs for the high-symmetry pathway workflow.

:class:`SymmetryPreviewDialog` shows what symmetry was detected and lets the
user retune the tolerance before committing to a run.
:class:`KPathDialog` is the single entry point for choosing a k-path: take
seekpath's full recommended path, or build a custom one from this lattice's
high-symmetry points -- ticking recommended segments, ticking other
point-to-point connections, or adding hand-picked segments, then ordering
them. Mirrors the workflow of the reference notebook's interactive k-path
selector (``Qvista-Phonopy_again.ipynb``, last cell), reimplemented with
native Qt widgets. Building does not close the dialog: the Brillouin zone,
the k-path preview and the KPOINTS download all appear in place, and stay
until the dialog is closed by hand.
"""

from __future__ import annotations

from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qvista.analyses.kpath import KPathResult, group_path_segments
from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.visualization.brillouin_plots import pretty_label

#: Tolerances worth trying, loosest first.
SYMPREC_PRESETS = (1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4)

# The look of these lives in qvista.gui.theme, and is re-exported here
# because this module is where the rest of the program has always
# imported it from. Nothing in a dialog should be choosing colours.
ERROR_STYLE = theme.ERROR_STYLE
WARNING_STYLE = theme.WARNING_STYLE

PRIMARY_BUTTON_STYLE = theme.PRIMARY_BUTTON_STYLE
INFO_BUTTON_STYLE = theme.INFO_BUTTON_STYLE
WARNING_BUTTON_STYLE = theme.WARNING_BUTTON_STYLE
SUCCESS_BUTTON_STYLE = theme.SUCCESS_BUTTON_STYLE


def _hline() -> QFrame:
    """A hairline rule, used to separate sections of a dialog."""

    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {theme.BORDER}; border: none;")

    return line


def _clear_layout(layout) -> None:
    """Remove and delete every widget/sub-layout ``layout`` holds."""

    while layout.count():

        item = layout.takeAt(0)

        widget = item.widget()

        if widget is not None:
            widget.deleteLater()
            continue

        sublayout = item.layout()

        if sublayout is not None:
            _clear_layout(sublayout)


# ------------------------------------------------------------
# Symmetry preview
# ------------------------------------------------------------


class SymmetryPreviewDialog(QDialog):
    """
    Show the detected symmetry and let the user adjust ``symprec``.

    The preview is recomputed whenever the tolerance changes, so the effect
    of loosening it is visible before a long calculation starts.
    """

    def __init__(
        self,
        preview_func: Callable[[float], Any],
        symprec: float,
        parent=None,
    ) -> None:
        """
        Parameters
        ----------
        preview_func
            Called with a tolerance, returns a
            :class:`~qvista.analyses.guards.SymmetryPreview`. Injected so
            this dialog stays independent of how the structure is obtained.
        symprec
            Starting tolerance.
        """

        super().__init__(parent)

        self.setWindowTitle("Symmetry check")
        self.setMinimumWidth(460)

        self._preview_func = preview_func

        self.symprec = symprec

        layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        # Tolerance
        # ----------------------------------------------------------

        form = QFormLayout()

        self.symprec_box = QDoubleSpinBox()
        self.symprec_box.setDecimals(5)
        self.symprec_box.setRange(0.00001, 1.0)
        self.symprec_box.setSingleStep(0.005)
        self.symprec_box.setValue(symprec)
        self.symprec_box.setSuffix(" Å")

        form.addRow("Symmetry tolerance", self.symprec_box)

        layout.addLayout(form)

        # ----------------------------------------------------------
        # Preset tolerances
        #
        # Symmetry is not a smooth function of the tolerance -- it jumps at
        # thresholds -- so stepping a spin box is a poor way to explore it.
        # These are the values worth trying, one click each.
        # ----------------------------------------------------------

        presets = QHBoxLayout()

        presets.addWidget(QLabel("Presets:"))

        for value in SYMPREC_PRESETS:

            button = QPushButton(f"{value:g}")
            button.setMaximumWidth(70)

            button.clicked.connect(
                partial(self.symprec_box.setValue, value)
            )

            presets.addWidget(button)

        presets.addStretch(1)

        layout.addLayout(presets)

        # ----------------------------------------------------------
        # Detected symmetry
        # ----------------------------------------------------------

        self.details = QLabel()
        self.details.setTextFormat(Qt.PlainText)

        layout.addWidget(self.details)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(WARNING_STYLE)

        layout.addWidget(self.warning)

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

        self.symprec_box.valueChanged.connect(self._refresh)

        self._refresh()

    # ------------------------------------------------------------

    def _refresh(self) -> None:
        """Recompute the preview at the current tolerance."""

        self.symprec = self.symprec_box.value()

        try:
            preview = self._preview_func(self.symprec)

        except Exception as exc:
            self.details.setText("")
            self.warning.setStyleSheet(ERROR_STYLE)
            self.warning.setText(str(exc))
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return

        self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)

        self.details.setText(
            "\n".join(
                f"{label:<21}: {value}"
                for label, value in preview.as_rows()
            )
        )

        self.warning.setStyleSheet(WARNING_STYLE)
        self.warning.setText("\n\n".join(preview.warnings))


# ------------------------------------------------------------
# High symmetry pathway
# ------------------------------------------------------------


class KPathDialog(QDialog):
    """
    Choose a high-symmetry k-path, build it, and inspect the result.

    Reimplements the reference notebook's interactive k-path selector
    (``Qvista-Phonopy_again.ipynb``, last cell) as native Qt widgets: tick
    FULL PATH for seekpath's recommended path -- which disables every other
    option, since they would have no effect -- or tick recommended
    segments / other point-to-point connections / hand-added segments and
    assign each a step in "Confirm Selection" -> "Build Ordered Path".

    Building does not close the dialog. It emits :attr:`path_requested`;
    the caller runs the analysis (on a worker thread, via the normal
    analysis pipeline) and calls :meth:`show_result` with the outcome,
    which previews the path and its KPOINTS right here and enables the
    download. The Brillouin zone itself is drawn full-size in the main
    window's Plots window rather than duplicated in this dialog. Nothing
    here closes on a build, so several builds can be tried before the
    dialog is closed by hand.
    """

    #: Emitted when the user has settled on a selection (Select ALL, or a
    #: fully ordered custom build) and it should be computed.
    path_requested = Signal()

    def __init__(
        self,
        points: dict[str, Any],
        recommended: Sequence[tuple[str, str]],
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.setWindowTitle("High Symmetry Pathway")
        fit_to_screen(self, 720, 640)

        self.points = dict(points)
        self.recommended = list(recommended)

        #: True means "use the recommended path as-is". Read once
        #: :attr:`path_requested` fires.
        self.use_recommended = True

        #: The ordered segments to build the path from, when
        #: `use_recommended` is False.
        self.custom_segments: list[tuple[str, str]] = []

        #: Points per segment, for the exported KPOINTS density.
        self.npoints = 20

        self._added_custom: list[tuple[str, str]] = []
        self._order_dropdowns: list[QComboBox] = []
        self._seg_label_map: dict[str, tuple[str, str]] = {}
        self._status_text = ""
        self._last_result: KPathResult | None = None

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)

        labels = sorted(self.points)

        # ----------------------------------------------------------
        # FULL PATH -- checking this disables every option below, since
        # they only matter for a custom build.
        # ----------------------------------------------------------

        self.select_all = QCheckBox("FULL PATH")
        self.select_all.setStyleSheet("font-weight: bold;")
        self.select_all.toggled.connect(self._toggle_full_path)

        layout.addWidget(self.select_all)

        self.selection_box = QWidget()
        selection_layout = QVBoxLayout(self.selection_box)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.selection_box)

        # ----------------------------------------------------------
        # Recommended path segments
        # ----------------------------------------------------------

        selection_layout.addWidget(
            _bold_label("Recommended path segments")
        )

        directed: list[tuple[str, str]] = []

        for start, end in self.recommended:
            directed.append((start, end))
            directed.append((end, start))

        self.recommended_checks: list[
            tuple[tuple[str, str], QCheckBox]
        ] = []

        for start, end in directed:

            box = QCheckBox(
                f"{pretty_label(start)}  →  {pretty_label(end)}"
            )

            self.recommended_checks.append(((start, end), box))
            selection_layout.addWidget(box)

        selection_layout.addWidget(_hline())

        # ----------------------------------------------------------
        # Other connections
        # ----------------------------------------------------------

        selection_layout.addWidget(
            _bold_label(
                "Other connections (not part of the recommended path)"
            )
        )

        recommended_pairs = {
            frozenset(pair) for pair in self.recommended
        }

        self.other_checks: list[tuple[tuple[str, str], QCheckBox]] = []

        for a, b in combinations(labels, 2):

            if frozenset((a, b)) in recommended_pairs:
                continue

            box = QCheckBox(
                f"{pretty_label(a)}  ↔  {pretty_label(b)}"
            )

            self.other_checks.append(((a, b), box))
            selection_layout.addWidget(box)

        selection_layout.addWidget(_hline())

        # ----------------------------------------------------------
        # Add a custom segment
        # ----------------------------------------------------------

        selection_layout.addWidget(_bold_label("Add a custom segment"))

        add_row = QHBoxLayout()

        add_row.addWidget(QLabel("From:"))

        self.custom_from = QComboBox()
        self.custom_from.addItems(labels)
        add_row.addWidget(self.custom_from)

        add_row.addWidget(QLabel("To:"))

        self.custom_to = QComboBox()
        self.custom_to.addItems(labels)
        add_row.addWidget(self.custom_to)

        add_button = QPushButton("Add custom segment")
        add_button.setStyleSheet(INFO_BUTTON_STYLE)
        add_button.clicked.connect(self._add_custom_segment)
        add_row.addWidget(add_button)

        clear_button = QPushButton("Clear custom")
        clear_button.setStyleSheet(WARNING_BUTTON_STYLE)
        clear_button.clicked.connect(self._clear_custom_segments)
        add_row.addWidget(clear_button)

        selection_layout.addLayout(add_row)

        self.custom_label = QLabel("(none)")
        self.custom_label.setWordWrap(True)

        selection_layout.addWidget(self.custom_label)

        layout.addWidget(_hline())

        # ----------------------------------------------------------
        # Points per segment
        # ----------------------------------------------------------

        points_row = QHBoxLayout()

        points_row.addWidget(QLabel("Pts/segment:"))

        self.npoints_box = QSpinBox()
        self.npoints_box.setRange(2, 200)
        self.npoints_box.setValue(20)

        points_row.addWidget(self.npoints_box)
        points_row.addStretch(1)

        layout.addLayout(points_row)

        # ----------------------------------------------------------
        # Confirm -> per-step ordering -> build
        # ----------------------------------------------------------

        confirm_button = QPushButton("1) Confirm Selection")
        confirm_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        confirm_button.clicked.connect(self._confirm_selection)

        layout.addWidget(confirm_button)

        self.order_layout = QVBoxLayout()

        layout.addLayout(self.order_layout)

        # ----------------------------------------------------------
        # Status / result text
        # ----------------------------------------------------------

        self.status_output = QPlainTextEdit()
        self.status_output.setReadOnly(True)
        self.status_output.setMinimumHeight(140)
        self.status_output.setStyleSheet(
            "font-family: Consolas, monospace;"
        )

        layout.addWidget(self.status_output)

        # The Brillouin zone itself is drawn full-size in the main window's
        # Plots window, not duplicated here -- one big picture rather than
        # two small ones.

        # ----------------------------------------------------------
        # Download
        # ----------------------------------------------------------

        download_row = QHBoxLayout()
        download_row.addStretch(1)

        self.download_button = QPushButton("Download KPOINTS")
        self.download_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._download_kpoints)

        download_row.addWidget(self.download_button)

        layout.addLayout(download_row)

        # ----------------------------------------------------------
        # Close -- nothing above is torn down until this is clicked.
        # ----------------------------------------------------------

        close_row = QHBoxLayout()
        close_row.addStretch(1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        close_row.addWidget(close_button)

        outer.addLayout(close_row)

        self._set_status(self._full_reference_text())

    # ------------------------------------------------------------
    # Building the path
    # ------------------------------------------------------------

    def _recommended_path_string(self) -> str:
        """Render the recommended path, branches separated by ``|``."""

        branches = group_path_segments(self.recommended)

        return "  |  ".join(
            " → ".join(pretty_label(label) for label in branch)
            for branch in branches
        )

    def _full_reference_text(self) -> str:
        """The always-visible reminder of what FULL PATH would use."""

        return (
            "Full recommended path (FULL PATH):\n"
            f"  {self._recommended_path_string()}"
        )

    def _toggle_full_path(self, checked: bool) -> None:
        """FULL PATH makes every other option irrelevant, so disable them."""

        self.selection_box.setEnabled(not checked)

        if checked:
            self._clear_order_area()

    def _set_status(self, message: str) -> None:
        """Replace the status text, keeping the last built result appended."""

        self._status_text = message

        if self._last_result is not None:
            message = f"{message}\n\n{self._describe_result(self._last_result)}"

        self.status_output.setPlainText(message)

    def _add_custom_segment(self) -> None:
        """Append the chosen From/To pair to the custom segment list."""

        start = self.custom_from.currentText()
        end = self.custom_to.currentText()

        if start == end:
            return

        self._added_custom.append((start, end))

        lines = ["Custom segments added:"]
        lines += [
            f"  {pretty_label(s)} → {pretty_label(e)}"
            for s, e in self._added_custom
        ]

        self.custom_label.setText("\n".join(lines))

    def _clear_custom_segments(self) -> None:
        """Remove every hand-added segment."""

        self._added_custom.clear()
        self.custom_label.setText("(none)")

    def _clear_order_area(self) -> None:
        """Remove the per-step dropdowns and the Build button, if shown."""

        self._order_dropdowns = []
        self._seg_label_map = {}

        _clear_layout(self.order_layout)

    def _confirm_selection(self) -> None:
        """
        Settle what is about to be built.

        "Select ALL" builds immediately. Otherwise this gathers whatever is
        ticked or added and opens one "Step N" dropdown per segment, so the
        traversal order can be set before building.
        """

        self.npoints = self.npoints_box.value()

        if self.select_all.isChecked():

            self._clear_order_area()
            self.use_recommended = True
            self._set_status(self._full_reference_text())

            self.path_requested.emit()
            return

        pending = [
            seg
            for seg, box in self.recommended_checks
            if box.isChecked()
        ]
        pending += [
            seg for seg, box in self.other_checks if box.isChecked()
        ]
        pending += list(self._added_custom)

        self._clear_order_area()

        if not pending:
            self._set_status(
                f"{self._full_reference_text()}\n\n"
                "No segments selected. Tick some boxes above, or add a "
                "custom one, then click 'Confirm Selection'."
            )
            return

        self._seg_label_map = {
            f"{pretty_label(s)} → {pretty_label(e)}": (s, e)
            for s, e in pending
        }

        step_labels = list(self._seg_label_map.keys())

        for index, label in enumerate(step_labels):

            row = QHBoxLayout()
            row.addWidget(QLabel(f"Step {index + 1}:"))

            box = QComboBox()
            box.addItems(step_labels)
            box.setCurrentIndex(index)

            row.addWidget(box)

            self._order_dropdowns.append(box)
            self.order_layout.addLayout(row)

        build_button = QPushButton("2) Build Ordered Path && Visualize")
        build_button.setStyleSheet(SUCCESS_BUTTON_STYLE)
        build_button.clicked.connect(self._build_ordered_path)

        self.order_layout.addWidget(build_button)

        self._set_status(
            f"{self._full_reference_text()}\n\n"
            f"{len(pending)} segment(s) selected. Set the traversal order "
            "above, then click 'Build Ordered Path & Visualize'."
        )

    def _build_ordered_path(self) -> None:
        """Read the per-step dropdowns and request the ordered build."""

        ordered = [
            self._seg_label_map[box.currentText()]
            for box in self._order_dropdowns
        ]

        used = [box.currentText() for box in self._order_dropdowns]

        message = self._full_reference_text()

        if len(set(used)) != len(used):
            message += (
                "\n\nWarning: the same segment was picked for more than "
                "one step (the repeat will just be walked again)."
            )

        self._set_status(message)

        self.use_recommended = False
        self.custom_segments = ordered

        self.path_requested.emit()

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    def _describe_result(self, result: KPathResult) -> str:
        """Render the built path, its coordinates, and a KPOINTS preview."""

        branches = result.branches()

        final_path = "  |  ".join(
            " → ".join(pretty_label(label) for label in branch)
            for branch in branches
        )

        lines = [f"Final path: {final_path}", ""]

        lines.append("Segments (fractional reciprocal coordinates):")

        for branch in branches:
            for start, end in zip(branch[:-1], branch[1:]):

                c1 = result.points[start]
                c2 = result.points[end]

                lines.append(
                    f"  {pretty_label(start):>3s}  "
                    f"{float(c1[0]):7.4f} {float(c1[1]):7.4f} "
                    f"{float(c1[2]):7.4f}"
                )
                lines.append(
                    f"  {pretty_label(end):>3s}  "
                    f"{float(c2[0]):7.4f} {float(c2[1]):7.4f} "
                    f"{float(c2[2]):7.4f}"
                )
                lines.append("")

        lines.append("--- KPOINTS preview ---")
        lines.append(result.to_kpoints())

        return "\n".join(lines)

    def show_result(self, result: KPathResult) -> None:
        """
        Show the built path's preview and enable its download.

        Called by the caller once the analysis (run on a worker thread)
        completes -- the dialog never runs it directly. Nothing here closes
        the dialog: several builds can be tried in a row. The Brillouin
        zone itself is drawn in the main window's Plots window, which the
        caller shows separately -- not duplicated here.
        """

        self._last_result = result

        self.status_output.setPlainText(
            f"{self._status_text}\n\n{self._describe_result(result)}"
        )

        self.download_button.setEnabled(True)

    def show_error(self, message: str) -> None:
        """Report a failed build without disturbing anything already shown."""

        self.status_output.setPlainText(
            f"{self._status_text}\n\nBuild failed: {message}"
        )

    def _download_kpoints(self) -> None:
        """Save the last built path as a VASP line-mode KPOINTS file."""

        if self._last_result is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Download KPOINTS",
            "KPOINTS",
            "KPOINTS File (KPOINTS);;All Files (*)",
        )

        if not filename:
            return

        try:
            Path(filename).write_text(
                self._last_result.to_kpoints(), encoding="utf-8"
            )

        except OSError as exc:
            QMessageBox.critical(
                self, "Download failed", f"Could not write {filename}: {exc}"
            )
            return

        QMessageBox.information(
            self, "Downloaded", f"KPOINTS written to {filename}"
        )


def _bold_label(text: str) -> QLabel:
    """A section heading, matching the reference notebook's ``<b>`` tags."""

    label = QLabel(text)
    label.setStyleSheet("font-weight: bold;")

    return label
