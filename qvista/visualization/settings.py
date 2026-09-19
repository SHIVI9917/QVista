"""
Visualization settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from qvista.visualization.bonding import BondRules


###############################################################################
# Enumerations
###############################################################################

class Representation(StrEnum):
    """Atomic representation."""

    BALL_STICK = "ball-stick"
    SPACE_FILL = "space-fill"
    POLYHEDRAL = "polyhedral"

class LabelMode(StrEnum):
    """Atom label style."""

    NONE = "none"

    INDEX = "index"

    SYMBOL = "symbol"

    SYMBOL_INDEX = "symbol_index"

    SYMBOL_GLOBAL = "symbol_global"

class ColorScheme(StrEnum):
    """Element color scheme."""

    CPK = "cpk"
    JMOL = "jmol"
    VESTA = "vesta"


class BondingMethod(StrEnum):
    """Bond generation algorithm."""

    COVALENT = "covalent"
    CUTOFF = "cutoff"


class Projection(StrEnum):
    """
    How the camera maps the scene onto the screen.

    Perspective is what the eye does: further atoms are drawn smaller,
    and parallel lines converge. It reads as depth, which is what makes
    a structure legible while it is being turned.

    Parallel keeps every distance to scale regardless of depth. Along a
    crystallographic axis that is the difference between a picture of a
    lattice and a measurement of one -- columns of atoms line up exactly
    when they are actually aligned, so a channel down the c axis is
    visibly open or visibly not. It is the projection a figure wants,
    and the one every crystallography package defaults to.
    """

    PERSPECTIVE = "perspective"
    PARALLEL = "parallel"


###############################################################################
# Boundary
###############################################################################

@dataclass(slots=True)
class Boundary:
    """
    Fractional display boundary.
    """

    x_minus: float = 0.0
    x_plus: float = 1.0

    y_minus: float = 0.0
    y_plus: float = 1.0

    z_minus: float = 0.0
    z_plus: float = 1.0


@dataclass(slots=True)
class Origin:
    """
    A fractional shift applied to the contents before they are drawn.

    The cell box stays where it is and the atoms move through it, coming
    back in the far side as they leave -- which is what makes it useful.
    A slab sitting against the bottom of its cell, with the vacuum above,
    can be walked up into the middle where it can be seen; an atom split
    across the cell boundary can be brought whole into view.

    It changes nothing about the structure. This is a shift of the
    origin the picture is drawn from, not an edit of the coordinates.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:

        return (self.x, self.y, self.z)

    @property
    def is_zero(self) -> bool:
        """True when nothing is being shifted."""

        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0


###############################################################################
# Viewer Settings
###############################################################################

@dataclass(slots=True)
class ViewerSettings:
    """Visualization settings."""

    # ==========================================================
    # Representation
    # ==========================================================

    representation: Representation = Representation.BALL_STICK

    # ==========================================================
    # Camera
    # ==========================================================

    # Parallel by default. A crystal is looked at to judge whether things
    # line up -- whether a channel runs clear down an axis, whether two
    # layers are registered -- and perspective makes that judgement
    # unreliable by drawing the far end of the cell smaller than the
    # near end. Every crystallography package defaults this way.
    projection: Projection = Projection.PARALLEL

    # ==========================================================
    # Colors
    # ==========================================================

    color_scheme: ColorScheme = ColorScheme.CPK

    # White, with the faintest cool tint at the floor so the spheres
    # have something to sit on rather than reading as stickers cut out
    # and pasted down. The window chrome around it stays dark; the
    # viewport is the one place a figure gets lifted out of the app and
    # into a document, and it wants a white ground for that.
    background_color: tuple[float, float, float] = (
        0.945,
        0.953,
        0.969,
    )

    background_top: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )

    # ==========================================================
    # Bond generation
    # ==========================================================

    bonding_method: BondingMethod = BondingMethod.COVALENT

    # Which element pairs are drawn as bonded, and out to what distance.
    # Empty is the plain covalent criterion for everything; see
    # :mod:`qvista.visualization.bonding` for why that is worth being
    # able to override per pair.
    bond_rules: BondRules = field(default_factory=BondRules)

    # ==========================================================
    # Boundary
    # ==========================================================

    boundary: Boundary = field(default_factory=Boundary)

    # Where the contents sit inside that boundary. See :class:`Origin`.
    origin: Origin = field(default_factory=Origin)

    # ==========================================================
    # Visibility
    # ==========================================================

    show_axes: bool = True
    show_cell: bool = True
    show_bonds: bool = True

    # ---------- NEW ----------

    show_polyhedra: bool = False

    show_labels: bool = False

    # ==========================================================
    # Polyhedra
    # ==========================================================

    polyhedron_elements: tuple[str, ...] = (
        "Co",
    )

    polyhedron_opacity: float = 0.35

    polyhedron_show_edges: bool = True

    # ==========================================================
    # Appearance
    # ==========================================================

    atom_radius_scale: float = 0.45

    bond_cylinder_radius: float = 0.10

    # Dark, because the viewport is white. This was a pale lavender
    # when the background was near-black; left as it was, the cell box
    # would now be drawn in a colour a shade off the paper it sits on.
    cell_color: tuple[float, float, float] = (
        0.24,
        0.28,
        0.38,
    )

    cell_line_width: float = 2.5

    # ==========================================================
    # Labels
    # ==========================================================

    label_mode: LabelMode = LabelMode.INDEX

    label_font_size: int = 12

    # Dark for the same reason as the cell box: near-white labels were
    # right over a near-black viewport and vanish over a white one.
    label_color: tuple[float, float, float] = (
        0.12,
        0.15,
        0.22,
    )
