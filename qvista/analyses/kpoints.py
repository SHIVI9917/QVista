"""
K-point mesh generation for VASP input.

Two ways to settle on a mesh: a target spacing in reciprocal space, or
explicit subdivisions typed in directly. Both end up as the same thing --
an ``(nx, ny, nz)`` grid -- rendered into an explicit, zero-shift KPOINTS
file.

The two conventions, and why both are here
------------------------------------------
"K-spacing 0.02" is ambiguous, and the two readings differ by a factor
of ``2*pi`` -- which is a factor of ``2*pi`` in the subdivisions along
every axis, and so a factor of about 250 in the number of k-points.

**VASP's KSPACING tag** is in inverse angstrom with the ``2*pi`` already
in it. VASP's own default is 0.5, and the wiki calls anything above that
a coarse mesh. Its rule is::

    N_i = max(1, ceil(|b_i| / KSPACING))

with ``|b_i|`` the reciprocal lattice vector length in the physics
convention -- the one that carries ``2*pi``, and the one pymatgen's
``reciprocal_lattice`` returns.

**The k-point resolution** used by VASPKIT and quoted in most papers is
in units of ``2*pi/angstrom``. Typical values run 0.06 (coarse) to 0.01
(fine), and the resulting spacing between k-points is ``2*pi`` times the
number quoted. Its rule is::

    N_i = max(1, round(|b_i| / (2*pi * resolution)))

Both are offered because both are what somebody means when they type a
number in, and neither can be inferred from the number itself: 0.02 is a
fine resolution and an absurd KSPACING, 0.5 is a normal KSPACING and a
useless resolution. Callers therefore say which they mean, and the
dialog says so on screen next to the value.
"""

from __future__ import annotations

import math

#: The factor between the two conventions, and between a crystallographic
#: reciprocal lattice vector and a physics one.
TWO_PI = 2.0 * math.pi

#: Resolution presets, in units of ``2*pi/angstrom``, tightest (densest,
#: most expensive) last. These are the numbers quoted in papers and used
#: by VASPKIT, not values for VASP's KSPACING tag.
RESOLUTION_PRESETS = (0.06, 0.05, 0.04, 0.03, 0.02, 0.01)

#: Presets for VASP's KSPACING tag, in inverse angstrom. VASP's own
#: default is 0.5.
KSPACING_PRESETS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15)

#: How a target spacing is meant. See the module docstring.
KP_RESOLUTION = "resolution"
VASP_KSPACING = "kspacing"

#: What each convention is called, and in what units, for a label.
CONVENTIONS: dict[str, str] = {
    KP_RESOLUTION: "K-resolution (2π/Å)",
    VASP_KSPACING: "KSPACING (1/Å)",
}

#: The two schemes exposed in the menu. These are exactly the strings
#: VASP expects on the third line of an explicit-mesh KPOINTS file.
GAMMA_SCHEME = "Gamma"
MP_SCHEME = "Monkhorst Pack"


def reciprocal_lengths(structure) -> tuple[float, float, float]:
    """
    ``|b_1|, |b_2|, |b_3|`` in inverse angstrom, carrying ``2*pi``.

    The physics convention, which is what both rules below are written
    against and what the drawn Brillouin zone is in.
    """

    return tuple(
        float(length)
        for length in structure.lattice.reciprocal_lattice.abc
    )


def grid_from_kspacing(
    structure,
    kspacing: float,
) -> tuple[int, int, int]:
    """
    Subdivisions for VASP's KSPACING tag, in inverse angstrom.

    ``N_i = max(1, ceil(|b_i| / KSPACING))`` -- VASP's own rule, ceiling
    and not rounding, so the mesh is never coarser than the number asks
    for.
    """

    if kspacing <= 0.0:
        raise ValueError("KSPACING has to be greater than zero.")

    return tuple(
        max(1, int(math.ceil(length / kspacing)))
        for length in reciprocal_lengths(structure)
    )


def grid_from_resolution(
    structure,
    resolution: float,
) -> tuple[int, int, int]:
    """
    Subdivisions for a k-point resolution in units of ``2*pi/angstrom``.

    ``N_i = max(1, round(|b_i| / (2*pi * resolution)))``. Typing 0.02
    therefore asks for k-points about ``2*pi * 0.02 = 0.126`` inverse
    angstrom apart, which is what the number means everywhere it is
    quoted.

    Rounding rather than ceiling, which is what VASPKIT does: at this
    resolution the difference is one subdivision, and rounding keeps a
    cell whose axes are nearly equal from coming out anisotropic.
    """

    if resolution <= 0.0:
        raise ValueError("The resolution has to be greater than zero.")

    target = TWO_PI * resolution

    return tuple(
        max(1, int(round(length / target)))
        for length in reciprocal_lengths(structure)
    )


def grid_for(
    structure,
    spacing: float,
    convention: str = KP_RESOLUTION,
) -> tuple[int, int, int]:
    """The mesh for ``spacing``, read in whichever convention is named."""

    if convention == VASP_KSPACING:
        return grid_from_kspacing(structure, spacing)

    if convention == KP_RESOLUTION:
        return grid_from_resolution(structure, spacing)

    raise ValueError(
        f"Unknown spacing convention {convention!r}. Use "
        f"{KP_RESOLUTION!r} or {VASP_KSPACING!r}."
    )


def achieved_spacing(
    structure,
    grid: tuple[int, int, int],
) -> tuple[float, float, float]:
    """
    The gap between neighbouring k-points along each axis, in 1/angstrom.

    What the mesh actually delivers, as against what was asked for. The
    two differ because the subdivisions are whole numbers, and on a long
    axis a single subdivision can be most of the difference -- which is
    worth showing rather than leaving the user to infer.
    """

    return tuple(
        length / max(1, int(count))
        for length, count in zip(reciprocal_lengths(structure), grid)
    )


def kpoints_content(nx: int, ny: int, nz: int, scheme: str) -> str:
    """Render an explicit, zero-shift automatic-mesh KPOINTS file."""

    return (
        "Automatic mesh\n"
        "0\n"
        f"{scheme}\n"
        f"{int(nx)} {int(ny)} {int(nz)}\n"
        "0 0 0\n"
    )


def mesh_points(
    grid: tuple[int, int, int],
    scheme: str = GAMMA_SCHEME,
):
    """
    Fractional coordinates of every k-point in the mesh, in ``[-0.5, 0.5)``.

    A Gamma-centred mesh puts a point at the origin and steps by ``1/N``;
    a Monkhorst-Pack mesh of even ``N`` straddles it, stepping by the
    same ``1/N`` from a half-step offset. That difference is the whole of
    what the scheme means, and it is only visible once the points are
    drawn, which is what this is for.

    Returned folded into the first zone, so the mesh sits around the
    origin rather than running off into the next cell.
    """

    import numpy as np

    axes = []

    for count in grid:

        count = max(1, int(count))

        steps = np.arange(count, dtype=float)

        if scheme == MP_SCHEME:
            # VASP's Monkhorst-Pack: (2n - N - 1) / 2N.
            coordinates = (2.0 * steps - count + 1.0) / (2.0 * count)
        else:
            coordinates = steps / count

        # Fold into [-0.5, 0.5): a mesh is periodic, and the copy around
        # the origin is the one inside the Brillouin zone.
        coordinates = (coordinates + 0.5) % 1.0 - 0.5

        axes.append(np.sort(coordinates))

    mesh = np.meshgrid(*axes, indexing="ij")

    return np.stack([axis.ravel() for axis in mesh], axis=1)
