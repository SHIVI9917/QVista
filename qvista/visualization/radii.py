"""
Atomic radii utilities.

Uses ASE's built-in atomic data.
"""

from __future__ import annotations

import math
import warnings

from ase.data import (
    atomic_numbers,
    covalent_radii,
    vdw_radii,
)

from qvista.visualization.settings import Representation


# ----------------------------------------------------------------------
# Covalent radius
# ----------------------------------------------------------------------

def covalent_radius(symbol: str) -> float:
    """
    Return the covalent radius (Å).
    """

    Z = atomic_numbers[symbol]

    r = covalent_radii[Z]

    if r <= 0:
        return 1.0

    if symbol == "H":
        r *= 1.50      # 20% larger


    return float(r)


# ----------------------------------------------------------------------
# van der Waals radius
# ----------------------------------------------------------------------

def vdw_radius(symbol: str) -> float:
    """
    Return the van der Waals radius (Å).

    pymatgen first, because ASE's table is NaN for forty-one elements --
    Sc through Ru, the whole lanthanide series, Hf through Ir, and most
    of the actinides. Falling back to the covalent radius for those, as
    this did, is not a smaller answer to the same question but an answer
    to a different one: covalent radii run roughly 0.6 of van der Waals,
    and mixing the two scales in one space-filling picture makes the
    elements that happen to have a vdW radius look enormous beside the
    ones that do not.

    The case that showed it was MgCoB. Boron has an ASE vdW radius of
    1.92 A and cobalt has none, so cobalt was drawn at its covalent
    1.26 A -- a metal rendered half the size of a metalloid. pymatgen
    has both (1.92 and 2.00), and the picture is right.

    Covalent is still the last resort. It is wrong in the way described
    above, but a missing radius has to become some number, and one that
    is too small is more obvious than one that is invented.
    """

    radius = _pymatgen_vdw(symbol)

    if radius is not None:
        return radius

    Z = atomic_numbers[symbol]

    r = vdw_radii[Z]

    # ASE stores unavailable values as NaN
    if math.isnan(r):
        return covalent_radius(symbol)

    if r <= 0:
        return covalent_radius(symbol)

    return float(r)


def _pymatgen_vdw(symbol: str) -> float | None:
    """
    pymatgen's van der Waals radius, or ``None`` if it has none either.

    Imported inside the function: pymatgen is slow to import and this
    module is pulled in by the viewer's own import chain, so paying for
    it at start-up would delay the window appearing. The result is
    cached per element, since this is called once per atom per redraw.
    """

    # The default has to be the sentinel, not None: a plain .get()
    # returns None for a missing key, which is indistinguishable from a
    # cached "pymatgen has no value for this" and would report every
    # element as unknown on the first call.
    cached = _VDW_CACHE.get(symbol, _UNKNOWN)

    if cached is not _UNKNOWN:
        return cached

    radius: float | None = None

    try:
        from pymatgen.core import Element

        # pymatgen warns for the handful of elements it has no value for
        # (lawrencium, and nothing else below Z=104). That is the case
        # this function exists to report, so the warning is noise.
        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            value = Element(symbol).van_der_waals_radius

        if value is not None:

            number = float(value)

            if number > 0 and not math.isnan(number):
                radius = number

    except Exception:
        # A symbol pymatgen does not recognise, or no pymatgen at all.
        # Either way the ASE table below is the answer.
        radius = None

    _VDW_CACHE[symbol] = radius

    return radius


#: Sentinel for "not looked up yet", distinct from a cached ``None``
#: meaning "looked up, and pymatgen has no value".
_UNKNOWN = object()

#: Element symbol to its pymatgen vdW radius, or ``None``.
_VDW_CACHE: dict[str, float | None] = {}


# ----------------------------------------------------------------------
# Generic radius selector
# ----------------------------------------------------------------------

def atomic_radius(
    symbol: str,
    representation: Representation,
) -> float:
    """
    Return the appropriate radius for the current
    visualization representation.

    Ball-and-stick  -> covalent radius
    Space-fill      -> van der Waals radius
    """

    if representation == Representation.SPACE_FILL:
        return vdw_radius(symbol)

    return covalent_radius(symbol)
