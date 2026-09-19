"""
Visualization control panel.

Five stacked sections, which together are taller than the dock they live
in on any ordinary window.  They therefore sit inside a scroll area: the
alternative is what this panel used to do, which was to report a rigid
929-pixel minimum and let the last section -- Color Scheme -- fall off
the bottom of the dock with no scrollbar and no way to reach it.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

#: How short the panel may get before the scroll area takes over. Low
#: enough that the window itself is never held open by this dock.
MINIMUM_PANEL_HEIGHT = 120


def _compact(layout, margins: tuple[int, int, int, int] | None = None) -> None:
    """
    Trim the gaps between a layout's widgets, in place.

    Only the spacing by default.  A group box's margins are not ours to
    take: the theme reserves padding at the top for the title, which is
    drawn on the frame itself, and a layout that claims that space back
    puts the first control straight through the lettering.
    """

    if layout is None:
        return

    layout.setSpacing(4)

    if margins is not None:
        layout.setContentsMargins(*margins)

from qvista.visualization.colors import (
    COLOR_SCHEME_NAMES,
    available_color_schemes,
)

from qvista.gui.icons import move_icon

from qvista.visualization.bonding import BondRules

from qvista.visualization.settings import (
    ViewerSettings,
    Representation,
    LabelMode,
    Projection,
)

class VisualizationPanel(QWidget):
    """Visualization settings panel."""

    settingsChanged = Signal(ViewerSettings)

    def __init__(self) -> None:
        super().__init__()

        self._settings = ViewerSettings()

        #: Which element pairs are drawn as bonded. Held here rather than
        #: read back out of a widget, because :meth:`_emit_settings`
        #: builds a fresh ViewerSettings on every change and anything it
        #: cannot read from a control is otherwise lost the next time
        #: any other control moves.
        self._bond_rules = BondRules()

        # The sections go into a scrolled page rather than straight into
        # the panel, so that a short window scrolls them instead of
        # cutting the last one off.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        # Width follows the dock, so the sections never need scrolling
        # sideways -- only ever down.
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()

        outer.addWidget(self.scroll)

        self.scroll.setWidget(page)

        self.setMinimumHeight(MINIMUM_PANEL_HEIGHT)

        layout = QVBoxLayout(page)

        # ==========================================================
        # Representation
        # ==========================================================

        representation = QGroupBox("Representation")
        rep_layout = QVBoxLayout(representation)

        self.ball_stick = QRadioButton("Ball && Stick")
        self.space_fill = QRadioButton("Space Fill")
        self.polyhedral = QRadioButton("Polyhedral")

        self.ball_stick.setChecked(True)

        rep_layout.addWidget(self.ball_stick)
        rep_layout.addWidget(self.space_fill)
        rep_layout.addWidget(self.polyhedral)

        layout.addWidget(representation)

        # ==========================================================
        # View
        # ==========================================================

        view = QGroupBox("View")
        view_layout = QGridLayout(view)

        self.x_button = QPushButton("X+")
        self.minus_x_button = QPushButton("X-")

        self.y_button = QPushButton("Y+")
        self.minus_y_button = QPushButton("Y-")

        self.z_button = QPushButton("Z+")
        self.minus_z_button = QPushButton("Z-")

        self.iso_button = QPushButton("Iso")
        self.reset_button = QPushButton("Reset")

        view_layout.addWidget(self.x_button, 0, 0)
        view_layout.addWidget(self.minus_x_button, 0, 1)

        view_layout.addWidget(self.y_button, 1, 0)
        view_layout.addWidget(self.minus_y_button, 1, 1)

        view_layout.addWidget(self.z_button, 2, 0)
        view_layout.addWidget(self.minus_z_button, 2, 1)

        view_layout.addWidget(self.iso_button, 3, 0)
        view_layout.addWidget(self.reset_button, 3, 1)

        # ----------------------------------------------------------
        # Drag
        # ----------------------------------------------------------

        # Across both columns, directly under the view buttons: it is a
        # mode rather than a one-shot move, and it stays on until it is
        # clicked off, so it should not sit in a row of things that fire
        # once.
        self.drag_button = QPushButton("  Drag")
        self.drag_button.setCheckable(True)
        self.drag_button.setCursor(Qt.PointingHandCursor)
        self.drag_button.setIcon(move_icon())
        self.drag_button.setIconSize(QSize(15, 15))
        self.drag_button.setToolTip(
            "Slide the whole scene with the left button instead of "
            "turning it. Panning is also on the middle button and on "
            "shift-drag, neither of which a trackpad has."
        )

        view_layout.addWidget(self.drag_button, 4, 0, 1, 2)

        # ----------------------------------------------------------
        # Projection
        # ----------------------------------------------------------

        # Checkable and mutually exclusive, so the pair reads as the
        # two states of one setting rather than two things to press.
        self.perspective_button = QPushButton("Perspective")
        self.parallel_button = QPushButton("Parallel")

        self.perspective_button.setToolTip(
            "Distant atoms drawn smaller, as the eye sees them. Easier "
            "to read depth while turning the structure."
        )

        self.parallel_button.setToolTip(
            "Every distance to scale regardless of depth. Columns of "
            "atoms line up exactly when they are actually aligned, "
            "which is what a figure down a crystal axis needs."
        )

        self.projection_group = QButtonGroup(self)
        self.projection_group.setExclusive(True)

        for button in (self.perspective_button, self.parallel_button):
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            self.projection_group.addButton(button)

        # Parallel is the default, matching ViewerSettings. See the
        # Projection enum for why a crystal wants it.
        self.parallel_button.setChecked(True)

        view_layout.addWidget(self.perspective_button, 5, 0)
        view_layout.addWidget(self.parallel_button, 5, 1)

        layout.addWidget(view)

        # ==========================================================
        # Boundary
        # ==========================================================


        boundary = QGroupBox("Boundary (Fractional)")
        grid = QGridLayout(boundary)

        # ----------------------------------------------------------
        # Spin boxes
        # ----------------------------------------------------------

        self.x_minus = QDoubleSpinBox()
        self.x_plus = QDoubleSpinBox()

        self.y_minus = QDoubleSpinBox()
        self.y_plus = QDoubleSpinBox()

        self.z_minus = QDoubleSpinBox()
        self.z_plus = QDoubleSpinBox()

        for widget, value in (
            (self.x_minus, 0.0),
            (self.y_minus, 0.0),
            (self.z_minus, 0.0),
            (self.x_plus, 1.0),
            (self.y_plus, 1.0),
            (self.z_plus, 1.0),
        ):
            widget.setRange(0.0, 10.0)
            widget.setDecimals(2)
            widget.setSingleStep(0.1)
            widget.setValue(value)

        # ----------------------------------------------------------
        # Layout
        # ----------------------------------------------------------

        grid.addWidget(QLabel("−X"), 0, 0)
        grid.addWidget(self.x_minus, 0, 1)
        grid.addWidget(QLabel("+X"), 0, 2)
        grid.addWidget(self.x_plus, 0, 3)

        grid.addWidget(QLabel("−Y"), 1, 0)
        grid.addWidget(self.y_minus, 1, 1)
        grid.addWidget(QLabel("+Y"), 1, 2)
        grid.addWidget(self.y_plus, 1, 3)

        grid.addWidget(QLabel("−Z"), 2, 0)
        grid.addWidget(self.z_minus, 2, 1)
        grid.addWidget(QLabel("+Z"), 2, 2)
        grid.addWidget(self.z_plus, 2, 3)

        layout.addWidget(boundary)

        # ==========================================================
        # Origin
        # ==========================================================

        origin = QGroupBox("Move Cell (Fractional)")
        origin_grid = QGridLayout(origin)

        self.origin_x = QDoubleSpinBox()
        self.origin_y = QDoubleSpinBox()
        self.origin_z = QDoubleSpinBox()

        for widget in (self.origin_x, self.origin_y, self.origin_z):
            # A full turn either way. Beyond one lattice vector the
            # picture repeats, so there is nothing further to reach.
            widget.setRange(-1.0, 1.0)
            widget.setDecimals(2)
            widget.setSingleStep(0.05)
            widget.setValue(0.0)

        self.origin_reset = QPushButton("Centre")
        self.origin_reset.setToolTip("Back to no shift.")

        origin_grid.addWidget(QLabel("X"), 0, 0)
        origin_grid.addWidget(self.origin_x, 0, 1)

        origin_grid.addWidget(QLabel("Y"), 1, 0)
        origin_grid.addWidget(self.origin_y, 1, 1)

        origin_grid.addWidget(QLabel("Z"), 2, 0)
        origin_grid.addWidget(self.origin_z, 2, 1)

        origin_grid.addWidget(self.origin_reset, 3, 0, 1, 2)

        origin.setToolTip(
            "Slide the contents through the cell. The box stays put and "
            "atoms re-enter the far side as they leave, so a slab lying "
            "against the bottom of its cell can be walked up into view."
        )

        layout.addWidget(origin)
        
        # ==========================================================
        # Display
        # ==========================================================

        display = QGroupBox("Display")
        display_layout = QVBoxLayout(display)

        self.show_cell = QCheckBox("Unit Cell")
        self.show_bonds = QCheckBox("Bonds")
        self.show_axes = QCheckBox("Axes")
        self.show_labels = QCheckBox("Labels")

        self.label_mode = QComboBox()

        self.label_mode.addItem(
            "Index",
            LabelMode.INDEX,
        )   

        self.label_mode.addItem(
            "Element",
            LabelMode.SYMBOL,
        )

        self.label_mode.addItem(
            "Element + Index",
            LabelMode.SYMBOL_INDEX,
        )

        self.label_mode.addItem(
            "Element + Global Index",
            LabelMode.SYMBOL_GLOBAL,
        )

        self.label_mode.setCurrentIndex(2)

        self.show_cell.setChecked(True)
        self.show_bonds.setChecked(True)
        self.show_axes.setChecked(True)

        display_layout.addWidget(self.show_cell)
        display_layout.addWidget(self.show_bonds)

        # Directly under the Bonds checkbox, because it is that checkbox
        # made specific: the box says whether bonds are drawn at all,
        # this says which of them.
        self.bond_types_button = QPushButton("Bond Types...")
        self.bond_types_button.setToolTip(
            "Choose which element pairs are bonded, and out to what "
            "distance."
        )

        display_layout.addWidget(self.bond_types_button)

        display_layout.addWidget(self.show_axes)
        display_layout.addWidget(self.show_labels)

        display_layout.addWidget(
            QLabel("Label Style")
        )

        display_layout.addWidget(
            self.label_mode
        )
        layout.addWidget(display)
        # ==========================================================
        # Color Scheme
        # ==========================================================

        colors = QGroupBox("Color Scheme")
        colors_layout = QVBoxLayout(colors)

        self.color_scheme = QComboBox()

        for scheme in available_color_schemes():
            self.color_scheme.addItem(
                COLOR_SCHEME_NAMES[scheme],
                scheme,
            )

        colors_layout.addWidget(self.color_scheme)

        layout.addWidget(colors)

        layout.addStretch()

        # ==========================================================
        # Connections
        # ==========================================================

        self.ball_stick.toggled.connect(self._emit_settings)
        self.space_fill.toggled.connect(self._emit_settings)
        self.polyhedral.toggled.connect(self._emit_settings)

        self.show_axes.stateChanged.connect(self._emit_settings)
        self.show_cell.stateChanged.connect(self._emit_settings)
        self.show_bonds.stateChanged.connect(self._emit_settings)
        self.show_labels.stateChanged.connect(self._emit_settings)
        self.label_mode.currentIndexChanged.connect(self._emit_settings)
        self.color_scheme.currentIndexChanged.connect(self._emit_settings)

        # Only the one that becomes checked needs to be heard; the other
        # fires the same signal on its way out and would emit twice.
        self.perspective_button.toggled.connect(self._on_projection)
        self.parallel_button.toggled.connect(self._on_projection)

        # Tightened last, so every section is already built: the five of
        # them at Qt's default spacing come to 929 pixels, which is
        # taller than the dock on an ordinary window. The scroll area
        # covers whatever is still over, and covers it at any window
        # size, which no amount of tightening can promise.
        _compact(layout, margins=(6, 4, 6, 4))

        for box in page.findChildren(QGroupBox):
            _compact(box.layout())

        for widget in (
            self.x_minus,
            self.x_plus,
            self.y_minus,
            self.y_plus,
            self.z_minus,
            self.z_plus,
            # Same treatment as the boundary: both are fractional
            # windows onto the same tiling, and a caller reading
            # settings() has to see either of them.
            self.origin_x,
            self.origin_y,
            self.origin_z,
        ):
            widget.valueChanged.connect(
                self._emit_settings
            )

        self.origin_reset.clicked.connect(self._reset_origin)

    def _reset_origin(self) -> None:
        """Put the contents back where the file has them."""

        for widget in (self.origin_x, self.origin_y, self.origin_z):
            widget.setValue(0.0)

    # ==========================================================
    # Public API
    # ==========================================================

    def settings(self) -> ViewerSettings:
        """Return current visualization settings."""

        return self._settings

    def set_settings(
        self,
        settings: ViewerSettings,
    ) -> None:
        """Replace current settings."""

        self._settings = settings

    def bond_rules(self) -> BondRules:
        """The bonding convention currently in force."""

        return self._bond_rules

    def set_bond_rules(self, rules: BondRules) -> None:
        """
        Replace the bonding convention and redraw.

        Called from the bond dialog on every change it makes, so the
        viewport follows the dialog rather than waiting for it to close.
        """

        self._bond_rules = rules

        self._emit_settings()

    # ==========================================================
    # Internal
    # ==========================================================

    def _on_projection(self, checked: bool) -> None:
        """
        One of the two projection buttons was clicked.

        An exclusive button group emits ``toggled`` twice for one click
        -- false from the button being released, then true from the one
        being pressed -- and acting on both would rebuild the scene for
        a state that lasted no time at all.
        """

        if checked:
            self._emit_settings()

    def _emit_settings(self) -> None:
        """Read widgets and emit ViewerSettings."""

        s = ViewerSettings()

        # ----------------------------------------------------------
        # Display (read first - representation below may override
        # show_bonds, since Space Fill and Polyhedral never show
        # explicit bonds regardless of the checkbox).
        # ----------------------------------------------------------

        s.show_axes = self.show_axes.isChecked()
        s.show_cell = self.show_cell.isChecked()
        s.show_bonds = self.show_bonds.isChecked()
        s.show_labels = self.show_labels.isChecked()

        # ----------------------------------------------------------
        # Representation
        # ----------------------------------------------------------

        if self.ball_stick.isChecked():

            s.representation = Representation.BALL_STICK

            s.show_polyhedra = False

        elif self.space_fill.isChecked():

            s.representation = Representation.SPACE_FILL

            s.show_polyhedra = False

            s.show_bonds = False

        else:

            s.representation = Representation.POLYHEDRAL

            s.show_polyhedra = True

            s.show_bonds = False

        # ----------------------------------------------------------
        # Camera
        # ----------------------------------------------------------

        s.projection = (
            Projection.PARALLEL
            if self.parallel_button.isChecked()
            else Projection.PERSPECTIVE
        )

        # ----------------------------------------------------------
        # Bonding
        # ----------------------------------------------------------

        s.bond_rules = self._bond_rules

        s.label_mode = self.label_mode.currentData()
        # Colors

        s.color_scheme = self.color_scheme.currentData()

        # Boundary

        s.boundary.x_minus = self.x_minus.value()
        s.boundary.x_plus = self.x_plus.value()

        s.boundary.y_minus = self.y_minus.value()
        s.boundary.y_plus = self.y_plus.value()

        s.boundary.z_minus = self.z_minus.value()
        s.boundary.z_plus = self.z_plus.value()

        # Origin

        s.origin.x = self.origin_x.value()
        s.origin.y = self.origin_y.value()
        s.origin.z = self.origin_z.value()

        self._settings = s

        self.settingsChanged.emit(s)
