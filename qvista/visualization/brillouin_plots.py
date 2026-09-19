"""
Brillouin zone figures.

Draws the first Brillouin zone -- the Wigner-Seitz cell of the reciprocal
lattice -- together with the reciprocal basis vectors, the high-symmetry
points, and the chosen k-path.

Everything is built from the *same* primitive cell the k-path came from.
That is the whole point: a Brillouin zone drawn from the input cell while
the k-points come from seekpath's standardised primitive cell puts the
points in the wrong place relative to the zone, which is the failure the
reference notebook describes as "only Gamma shows up".

Pure matplotlib, so the figure works identically in the embedded Qt canvas,
in a headless export, and in tests.
"""

from __future__ import annotations

from typing import Any

#: Rendered instead of the literal label "GAMMA".
GAMMA_SYMBOL = "Γ"

#: Colours for the reciprocal basis vectors b1, b2, b3.
BASIS_COLOURS = ("#c0392b", "#27ae60", "#2980b9")

#: Cycled across the path branches.
BRANCH_COLOURS = (
    "#c0392b",
    "#d68910",
    "#1e8449",
    "#6c3483",
    "#17202a",
)


def pretty_label(label: str) -> str:
    """Render a point label for display, with Greek gamma and subscripts."""

    if label == "GAMMA":
        return GAMMA_SYMBOL

    for suffix in ("_1", "_2", "_3", "_4"):
        if label.endswith(suffix):
            digit = "₁₂₃₄"[int(suffix[1]) - 1]
            return f"{label[:-2]}{digit}"

    return label


def brillouin_facets(reciprocal_lattice) -> list:
    """
    Return the polygonal facets of the first Brillouin zone.

    The zone is the Wigner-Seitz cell of the reciprocal lattice, so this is
    a direct pymatgen call on a lattice built from the reciprocal vectors.
    """

    from pymatgen.core import Lattice

    return Lattice(reciprocal_lattice).get_wigner_seitz_cell()


def draw_brillouin_zone(
    figure,
    result,
    show_basis: bool = True,
    show_path: bool = True,
    title: str = "",
) -> None:
    """
    Draw the Brillouin zone of ``result`` onto ``figure``.

    ``result`` is a :class:`~qvista.analyses.kpath.KPathResult`.
    """

    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    figure.clear()

    if result.primitive is None:
        raise ValueError(
            "The k-path carries no primitive cell, so its Brillouin zone "
            "cannot be drawn."
        )

    basis = np.asarray(result.primitive.reciprocal_lattice, dtype=float)

    axes = figure.add_subplot(111, projection="3d")

    # ----------------------------------------------------------
    # The zone itself
    # ----------------------------------------------------------

    edges = []

    for facet in brillouin_facets(basis):

        points = [np.asarray(point, dtype=float) for point in facet]

        # Close the polygon so every edge is drawn once.
        for start, end in zip(points, points[1:] + points[:1]):
            edges.append((start, end))

    axes.add_collection3d(
        Line3DCollection(
            edges,
            colors="#95a5a6",
            linewidths=0.9,
        )
    )

    # ----------------------------------------------------------
    # Reciprocal basis vectors
    # ----------------------------------------------------------

    if show_basis:

        for vector, colour, name in zip(
            basis,
            BASIS_COLOURS,
            ("b₁", "b₂", "b₃"),
        ):
            axes.quiver(
                0.0,
                0.0,
                0.0,
                vector[0],
                vector[1],
                vector[2],
                color=colour,
                linewidth=1.4,
                arrow_length_ratio=0.12,
            )

            axes.text(
                vector[0],
                vector[1],
                vector[2],
                f"  {name}",
                color=colour,
                fontsize=9,
            )

    # ----------------------------------------------------------
    # High-symmetry points
    # ----------------------------------------------------------

    cartesian = result.cartesian_points()

    if cartesian:

        coordinates = np.asarray(list(cartesian.values()), dtype=float)

        axes.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            s=18,
            color="#2c3e50",
            depthshade=False,
        )

        for label, point in cartesian.items():
            axes.text(
                point[0],
                point[1],
                point[2],
                f"  {pretty_label(label)}",
                fontsize=9,
                color="#2c3e50",
            )

    # ----------------------------------------------------------
    # The path
    # ----------------------------------------------------------

    if show_path and cartesian:

        for index, branch in enumerate(result.branches()):

            colour = BRANCH_COLOURS[index % len(BRANCH_COLOURS)]

            for start, end in zip(branch[:-1], branch[1:]):

                if start not in cartesian or end not in cartesian:
                    continue

                head = cartesian[start]
                tail = cartesian[end]

                delta = tail - head

                # An arrow rather than a line: the traversal direction is
                # part of what the path means.
                axes.quiver(
                    head[0],
                    head[1],
                    head[2],
                    delta[0],
                    delta[1],
                    delta[2],
                    color=colour,
                    linewidth=2.0,
                    arrow_length_ratio=0.18,
                )

    # ----------------------------------------------------------
    # Framing
    # ----------------------------------------------------------

    _equalise(axes, basis)

    axes.set_xlabel("$k_x$ (1/Å)")
    axes.set_ylabel("$k_y$ (1/Å)")
    axes.set_zlabel("$k_z$ (1/Å)")

    if title:
        axes.set_title(title)

    # ----------------------------------------------------------
    # Caption: which path is actually drawn above, in the same colours
    # as the arrows. Without this the zone alone does not say whether the
    # path shown is seekpath's recommendation or a hand-built one.
    # ----------------------------------------------------------

    if show_path and cartesian:

        source = "custom path" if result.is_custom else "recommended path (seekpath)"

        caption = f"{source}: {result.path_string()}"

        if len(caption) > 140:
            caption = caption[:137] + "..."

        figure.text(
            0.5,
            0.02,
            caption,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#2c3e50",
        )

        figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    else:
        figure.tight_layout()


def _equalise(axes, basis) -> None:
    """
    Give the three axes the same scale.

    Without this the zone is stretched and the geometry is unreadable --
    a cubic zone has to look cubic.
    """

    import numpy as np

    extent = float(np.abs(np.asarray(basis, dtype=float)).sum(axis=0).max())

    limit = 0.62 * extent

    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_zlim(-limit, limit)

    try:
        axes.set_box_aspect((1.0, 1.0, 1.0))

    except Exception:
        # Older matplotlib without set_box_aspect on 3D axes.
        pass
