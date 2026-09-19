"""
COHP, COOP and COBI curves.

Drawn the way the literature draws them: energy up the vertical axis with
the Fermi level at zero, and the quantity across the horizontal one with
bonding to the right.  For COHP that means plotting ``-COHP``, since its
convention is that a bond is negative; COOP and COBI are already
positive for bonding and are drawn as they come.
:mod:`qvista.services.cohp` decides which, so the axis label and the sign
can never disagree.

The integrated curve is the same thing accumulated from the bottom of the
valence band up, so its value where it crosses the Fermi level is the
bond strength quoted in the table.

Every curve is labelled and legended, and moving the cursor over one
names the bond it belongs to along with its length, its strength, and
the value under the cursor.  A dozen Co-O curves are indistinguishable
by colour alone, so pointing at one is the only practical way to ask
which is which.

Smoothing
---------
A COHP comes off a finite k-mesh sampled onto a finite energy grid, so
what should be a smooth resonance arrives with a visible corner at the
top of the peak.  Every curve here is therefore drawn through a cubic
B-spline, the same one the rest of QVista's figures go through.

That is interpolation, not broadening: the spline passes through every
calculated point, so a peak keeps its height and its position and the
area under the curve -- which *is* the bond strength quoted in the table
beside it -- is unchanged.  Only the number of points the line is drawn
with between the calculated ones goes up.

There used to be a Gaussian smearing control here instead, with a sigma
in eV.  Convolution moves the data: it lowers peaks, fills valleys and
can merge two nearby states into one, and the figure then claims a
single resonance where the calculation found two.  It is the right tool
for saying "this run does not resolve structure finer than sigma" and
the wrong one for taking the corners off a line.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from qvista.gui.dialogs.side_window import PREFERRED_SIZE, SideWindow
from qvista.gui.widgets.plot_panel import PlotPanel
from qvista.services import cohp as ch
from qvista.visualization.smoothing import spline

#: How many separate curves are worth drawing at once before the figure
#: turns into a hairball and the summed one is the only useful view.
MAX_SEPARATE = 12

BONDING = "#36cfc2"
ANTIBONDING = "#ef6076"
INTEGRAL = "#f2a65a"
AVERAGE = "#9ba5be"

#: How near the cursor has to be to a curve, as a fraction of the panel's
#: width, before it counts as pointing at it.  Loose enough that the
#: curve does not have to be hit exactly, tight enough that pointing at
#: empty space names nothing.
HOVER_TOLERANCE = 0.05

def smooth(energies, values):
    """
    A curve resampled through a B-spline, ready to draw.

    Returns ``(energies, values)`` together, because the spline puts
    both on a denser grid than the file's and drawing one against the
    other's original axis would shear the curve.

    The energy axis is what is splined against, not the value: a COHP is
    a function of energy, and the same two arrays are handed back
    untouched whenever the grid is too short or too irregular for a
    cubic to be trustworthy.
    """

    energies = np.asarray(energies, dtype=float)
    values = np.asarray(values, dtype=float)

    if energies.shape != values.shape:
        return energies, values

    return spline(energies, values)


class CohpPlotWindow(SideWindow):
    """One window per set of bonds, replotted from inside."""

    def __init__(
        self,
        result: ch.CohpResult,
        bonds: list[ch.Bond],
        heading: str = "",
        parent=None,
    ) -> None:

        super().__init__(f"QVista - {result.kind.name}", parent, PREFERRED_SIZE)

        self.result = result
        self.bonds = [bond for bond in bonds if bond.signed_curve is not None]

        if not self.bonds and result.average is None:
            raise ValueError("None of those bonds has a curve to draw.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.panel = PlotPanel(self)

        layout.addLayout(self._build_controls())
        layout.addWidget(self.panel, 1)

        self.heading = heading

        #: Every curve currently drawn, as (axes, energies, values, text).
        #: Rebuilt on each draw and read by the hover handler.
        self._traces: list[tuple[object, object, object, str]] = []

        #: The floating label, one per axes, made on demand.
        self._tips: dict[object, object] = {}

        self.panel.canvas.mpl_connect(
            "motion_notify_event", self._on_hover
        )

        self._draw()

    # ------------------------------------------------------------

    def _build_controls(self) -> QHBoxLayout:

        row = QHBoxLayout()
        row.setSpacing(10)

        row.addWidget(QLabel("Show"))

        self.mode = QComboBox()
        self.mode.addItem(f"{self.result.kind.curve_label}", "curve")
        self.mode.addItem(f"Integrated ({self.result.kind.strength_label})", "integrated")
        self.mode.addItem("Both", "both")
        self.mode.setCurrentIndex(2)
        self.mode.currentIndexChanged.connect(lambda _: self._draw())

        row.addWidget(self.mode)

        self.separate = QCheckBox("Per bond")
        self.separate.setToolTip(
            f"Draw each bond on its own instead of adding them up. "
            f"Capped at {MAX_SEPARATE} curves."
        )
        self.separate.toggled.connect(lambda _: self._draw())

        row.addWidget(self.separate)

        self.with_average = QCheckBox("Cell average")
        self.with_average.setEnabled(self.result.average is not None)
        self.with_average.toggled.connect(lambda _: self._draw())

        row.addWidget(self.with_average)

        row.addWidget(QLabel("Energy"))

        self.low = QDoubleSpinBox()
        self.low.setRange(-100.0, 0.0)
        self.low.setValue(-10.0)
        self.low.setSuffix(" eV")
        self.low.valueChanged.connect(lambda _: self._draw())

        row.addWidget(self.low)

        self.high = QDoubleSpinBox()
        self.high.setRange(0.0, 100.0)
        self.high.setValue(6.0)
        self.high.setSuffix(" eV")
        self.high.valueChanged.connect(lambda _: self._draw())

        row.addWidget(self.high)

        row.addStretch(1)

        self.caption = QLabel("")

        row.addWidget(self.caption)

        return row

    # ------------------------------------------------------------

    def _draw(self) -> None:

        figure = self.panel.figure
        figure.clear()

        # Both are rebuilt below; the artists they referred to went with
        # the figure and following them now would crash.
        self._traces = []
        self._tips = {}

        mode = self.mode.currentData()

        axes = (
            figure.subplots(1, 2, sharey=True)
            if mode == "both"
            else [figure.subplots(1, 1)]
        )

        if mode in ("curve", "both"):
            self._draw_curve(axes[0], integrated=False)

        if mode == "integrated":
            self._draw_curve(axes[0], integrated=True)

        if mode == "both":
            self._draw_curve(axes[1], integrated=True)

        for axis in axes:

            axis.set_ylim(self.low.value(), self.high.value())
            axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
            axis.axvline(0.0, color="0.4", linewidth=0.8)

        axes[0].set_ylabel("E - E$_F$  (eV)")

        if self.heading:
            figure.suptitle(self.heading, fontsize=10)

        figure.tight_layout()

        self.panel.refresh()

        self.caption.setText(
            f"{len(self.bonds)} bond(s)"
            + (
                f"   total {self.result.kind.strength_label} "
                f"{sum(b.strength for b in self.bonds):.3f} "
                f"{self.result.kind.unit}"
                if self.bonds
                else ""
            )
        )

    def _draw_curve(self, axis, integrated: bool) -> None:
        """One panel: either the curve or its running integral."""

        kind = self.result.kind

        label = (
            f"Integrated {kind.strength_label}"
            if integrated
            else kind.curve_label
        )

        if self.bonds:

            if self.separate.isChecked():

                shown = self.bonds[:MAX_SEPARATE]

                for bond in shown:

                    values = (
                        bond.signed_integrated
                        if integrated
                        else bond.signed_curve
                    )

                    if values is None:
                        continue

                    energies, values = smooth(bond.energies, values)

                    axis.plot(
                        values,
                        energies,
                        linewidth=1.0,
                        label=bond.label(),
                    )

                    self._traces.append(
                        (axis, energies, values, self._describe(bond))
                    )

            else:

                pack = ch.summed_curve(self.bonds)

                if pack is not None:

                    energies, (curve, running) = pack

                    energies, values = smooth(
                        energies, running if integrated else curve
                    )

                    total = sum(b.strength for b in self.bonds)

                    summed_label = (
                        f"sum of {len(self.bonds)} bond(s)"
                        if len(self.bonds) > 1
                        else self.bonds[0].label()
                    )

                    axis.plot(
                        values, energies,
                        color=INTEGRAL if integrated else BONDING,
                        linewidth=1.2,
                        label=summed_label,
                    )

                    self._traces.append(
                        (
                            axis,
                            energies,
                            values,
                            self._describe(self.bonds[0])
                            if len(self.bonds) == 1
                            else (
                                f"{summed_label}\n"
                                f"{kind.strength_label} {total:.3f} "
                                f"{kind.unit}"
                            ),
                        )
                    )

                    # Filling the two sides in different colours is what
                    # makes a COHP plot readable at a glance: bonding to
                    # the right of zero, antibonding to the left.  The
                    # fills carry the legend entry that says which is
                    # which, so the reader never has to recall the
                    # convention.
                    if not integrated:

                        axis.fill_betweenx(
                            energies, 0, values,
                            where=values >= 0,
                            color=BONDING, alpha=0.35,
                            label="bonding",
                        )
                        axis.fill_betweenx(
                            energies, 0, values,
                            where=values < 0,
                            color=ANTIBONDING, alpha=0.35,
                            label="antibonding",
                        )

        if self.with_average.isChecked() and self.result.average is not None:

            values = (
                self.result.average_integrated
                if integrated
                else self.result.average
            )

            if values is not None:

                if kind.flip:
                    values = -values

                energies, values = smooth(self.result.energies, values)

                axis.plot(
                    values,
                    energies,
                    color=AVERAGE,
                    linewidth=1.0,
                    linestyle=":",
                    label="cell average",
                )

                self._traces.append(
                    (
                        axis,
                        energies,
                        values,
                        "cell average\nevery interaction in the run",
                    )
                )

        axis.set_xlabel(
            f"{label}" + (f"  ({kind.unit})" if kind.unit else "")
        )

        capped = (
            self.separate.isChecked() and len(self.bonds) > MAX_SEPARATE
        )

        if capped:
            axis.set_title(
                f"{MAX_SEPARATE} of {len(self.bonds)} bonds",
                fontsize=8,
                color="0.5",
            )

        elif not integrated:
            # Only worth saying where there is no bonding/antibonding
            # fill already saying it in the legend.
            axis.set_title(
                "bonding to the right", fontsize=8, color="0.5"
            )

        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7, loc="lower right", framealpha=0.85)

    # ------------------------------------------------------------
    # Pointing at a curve
    # ------------------------------------------------------------

    def _on_hover(self, event) -> None:
        """
        Name whichever curve the cursor is nearest.

        Matched along the energy axis rather than by distance in the
        plane: energy is the axis both panels share and the one the
        reader is already thinking in, so "which curve is furthest right
        at this energy" is the question being asked when they point.
        """

        if event.inaxes is None or event.xdata is None:
            self._hide_tips()
            return

        traces = [
            trace for trace in self._traces if trace[0] is event.inaxes
        ]

        if not traces:
            self._hide_tips()
            return

        left, right = event.inaxes.get_xlim()

        span = abs(right - left)

        if span <= 0.0:
            return

        best = None
        best_gap = None

        for _, energies, values, text in traces:

            energies = np.asarray(energies, dtype=float)
            values = np.asarray(values, dtype=float)

            if energies.size == 0 or event.ydata is None:
                continue

            # Outside the curve's own energy range np.interp would clamp
            # to an endpoint and report a hit that is not there.
            if not energies.min() <= event.ydata <= energies.max():
                continue

            at_cursor = float(np.interp(event.ydata, energies, values))

            gap = abs(at_cursor - event.xdata)

            if best_gap is None or gap < best_gap:
                best_gap = gap
                best = (text, at_cursor)

        if best is None or best_gap > span * HOVER_TOLERANCE:
            self._hide_tips()
            return

        text, at_cursor = best

        self._show_tip(
            event.inaxes,
            at_cursor,
            event.ydata,
            f"{text}\nE {event.ydata:+.2f} eV     here {at_cursor:+.3f}",
        )

    def _show_tip(self, axis, x: float, y: float, text: str) -> None:
        """Put the floating label on the curve, and repaint."""

        tip = self._tips.get(axis)

        if tip is None:

            tip = axis.annotate(
                "",
                xy=(0.0, 0.0),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=7,
                color="#f5f7fa",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#20242e",
                    "edgecolor": "#5a6274",
                    "alpha": 0.95,
                },
                annotation_clip=False,
                zorder=20,
            )

            self._tips[axis] = tip

        tip.xy = (x, y)
        tip.set_text(text)
        tip.set_visible(True)

        self.panel.canvas.draw_idle()

    def _hide_tips(self) -> None:
        """Take the label down when the cursor leaves every curve."""

        changed = False

        for tip in self._tips.values():

            if tip.get_visible():
                tip.set_visible(False)
                changed = True

        if changed:
            self.panel.canvas.draw_idle()

    def _describe(self, bond) -> str:
        """What the tooltip says about one bond."""

        kind = self.result.kind

        return (
            f"{bond.label()}\n"
            f"{bond.length:.3f} A     "
            f"{kind.strength_label} {bond.strength:.3f} {kind.unit}"
        )
