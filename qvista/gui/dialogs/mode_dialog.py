"""
Phonon mode animation dialog.

Lets the user pick a wavevector and a band, then plays that normal mode in
the main 3D viewer: the atoms oscillate along the eigenvector, which makes
the character of a mode obvious in a way a dispersion curve is not.

The frames are precomputed once when a mode is selected, so playback only
swaps positions and stays smooth.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from qvista.gui import theme
from qvista.analyses.phonon_modes import (
    DEFAULT_AMPLITUDE,
    DEFAULT_FRAMES,
    PhononMode,
    animation_frames,
    modes_at_qpoint,
)

#: Milliseconds between frames. 20 frames a second is smooth enough for
#: an oscillation and slow enough that a full scene redraw keeps up.
FRAME_INTERVAL = 50

WARNING_STYLE = theme.WARNING_STYLE


class ModeAnimationDialog(QDialog):
    """Pick a phonon mode and animate it in the viewer."""

    def __init__(
        self,
        phonon,
        points: dict[str, Any],
        on_frame: Callable[[Any], None],
        on_stop: Callable[[], None],
        parent=None,
    ) -> None:
        """
        Parameters
        ----------
        phonon
            The phonopy object carrying force constants.
        points
            High-symmetry label to fractional coordinate.
        on_frame
            Called with a ``(natoms, 3)`` Cartesian array per frame.
        on_stop
            Called when playback stops, to restore the original structure.
        """

        super().__init__(parent)

        self.setWindowTitle("Phonon mode animation")
        self.setMinimumWidth(520)

        self.phonon = phonon
        self.points = dict(points)
        self._on_frame = on_frame
        self._on_stop = on_stop

        self.modes: list[PhononMode] = []
        self.frames: list[Any] = []
        self._frame_index = 0

        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_INTERVAL)
        self.timer.timeout.connect(self._advance)

        layout = QVBoxLayout(self)

        # ----------------------------------------------------------
        # Wavevector
        # ----------------------------------------------------------

        form = QFormLayout()

        self.qpoint_box = QComboBox()

        for label in sorted(self.points):
            coords = self.points[label]
            self.qpoint_box.addItem(
                f"{label}   ({coords[0]:.3f}, {coords[1]:.3f}, "
                f"{coords[2]:.3f})",
                label,
            )

        form.addRow("Wavevector", self.qpoint_box)

        self.amplitude_box = QDoubleSpinBox()
        self.amplitude_box.setRange(0.02, 2.0)
        self.amplitude_box.setSingleStep(0.05)
        self.amplitude_box.setValue(DEFAULT_AMPLITUDE)
        self.amplitude_box.setSuffix(" Å")

        form.addRow("Amplitude", self.amplitude_box)

        layout.addLayout(form)

        # ----------------------------------------------------------
        # Modes
        # ----------------------------------------------------------

        layout.addWidget(QLabel("Modes at this wavevector:"))

        self.mode_list = QListWidget()

        layout.addWidget(self.mode_list)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(WARNING_STYLE)
        self.warning.setVisible(False)

        layout.addWidget(self.warning)

        # ----------------------------------------------------------
        # Transport
        # ----------------------------------------------------------

        transport = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle)
        self.play_button.setEnabled(False)

        transport.addWidget(self.play_button)

        self.export_button = QPushButton("Export trajectory...")
        self.export_button.setEnabled(False)

        transport.addWidget(self.export_button)

        transport.addStretch(1)

        layout.addLayout(transport)

        # ----------------------------------------------------------
        # Close
        # ----------------------------------------------------------

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        # ----------------------------------------------------------

        self.qpoint_box.currentIndexChanged.connect(self._load_modes)
        self.mode_list.currentRowChanged.connect(self._select_mode)
        self.amplitude_box.valueChanged.connect(self._rebuild_frames)

        self._load_modes()

    # ------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------

    @property
    def current_mode(self) -> PhononMode | None:
        """The mode currently selected, if any."""

        row = self.mode_list.currentRow()

        if 0 <= row < len(self.modes):
            return self.modes[row]

        return None

    def _load_modes(self) -> None:
        """Diagonalise at the selected wavevector."""

        self.stop()

        label = self.qpoint_box.currentData()

        if label is None:
            return

        try:
            self.modes = modes_at_qpoint(
                self.phonon,
                self.points[label],
                label=label,
            )

        except Exception as exc:
            self.modes = []
            self.warning.setText(f"Could not compute modes: {exc}")
            self.warning.setVisible(True)
            return

        self.mode_list.clear()

        for mode in self.modes:

            item = QListWidgetItem(mode.describe())

            if mode.is_imaginary:
                item.setForeground(Qt.red)

            self.mode_list.addItem(item)

        if self.modes:
            self.mode_list.setCurrentRow(0)

    def _select_mode(self, row: int) -> None:
        """Precompute the frames for the newly selected mode."""

        self.stop()

        mode = self.current_mode

        if mode is None:
            self.play_button.setEnabled(False)
            self.export_button.setEnabled(False)
            return

        self._rebuild_frames()

        self.play_button.setEnabled(True)
        self.export_button.setEnabled(True)

        if mode.is_imaginary:
            self.warning.setText(
                "This mode is imaginary: the structure lowers its energy by "
                "distorting along it. The animation shows the distortion "
                "direction, but the motion is not a real oscillation."
            )
            self.warning.setVisible(True)
        else:
            self.warning.setVisible(False)

    def _rebuild_frames(self) -> None:
        """Regenerate frames, e.g. after the amplitude changed."""

        import numpy as np

        mode = self.current_mode

        if mode is None:
            return

        primitive = self.phonon.primitive

        positions = np.asarray(primitive.positions, dtype=float)

        self.frames = animation_frames(
            positions,
            mode.eigenvector,
            primitive.masses,
            frames=DEFAULT_FRAMES,
            amplitude=self.amplitude_box.value(),
        )

        self._frame_index = 0

    # ------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------

    def _toggle(self) -> None:
        """Start or stop playback."""

        if self.timer.isActive():
            self.stop()
        else:
            self.play()

    def play(self) -> None:
        """Begin animating the selected mode."""

        if not self.frames:
            return

        self.play_button.setText("Stop")
        self.timer.start()

    def stop(self) -> None:
        """Stop animating and restore the original structure."""

        if self.timer.isActive():
            self.timer.stop()

        self.play_button.setText("Play")

        self._on_stop()

    def _advance(self) -> None:
        """Push the next frame to the viewer."""

        if not self.frames:
            return

        self._on_frame(self.frames[self._frame_index])

        self._frame_index = (self._frame_index + 1) % len(self.frames)

    # ------------------------------------------------------------

    def reject(self) -> None:
        """Stop playback before closing."""

        self.stop()

        super().reject()

    def closeEvent(self, event) -> None:
        """Stop playback if the window is closed from its title bar."""

        self.stop()

        super().closeEvent(event)
