"""
Default rules for coordination polyhedra.

These rules are heuristic defaults intended for visualization.
They work well for most inorganic crystal structures and can
be overridden later by the user.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Default cation elements (polyhedron centers)
# ----------------------------------------------------------------------

DEFAULT_CATIONS = {

    # Hydrogen
    "H",

    # Alkali metals
    "Li", "Na", "K", "Rb", "Cs", "Fr",

    # Alkaline earth metals
    "Be", "Mg", "Ca", "Sr", "Ba", "Ra",

    # Group 3
    "Sc", "Y",

    # Lanthanides
    "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",

    # Actinides
    "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",

    # 3d transition metals
    "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn",

    # 4d transition metals
    "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd",

    # 5d transition metals
    "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg",

    # p-block metals
    "Al", "Ga", "In", "Tl",
    "Sn", "Pb", "Bi",
}

# ----------------------------------------------------------------------
# Default anion / ligand elements (polyhedron vertices)
# ----------------------------------------------------------------------

DEFAULT_ANIONS = {

    # Boron
    "B",

    # Carbon
    "C",

    # Nitrogen group (pnictogens)
    "N", "P", "As", "Sb", "Bi",

    # Chalcogens
    "O", "S", "Se", "Te", "Po",

    # Halogens
    "F", "Cl", "Br", "I", "At",

    # Hydrogen (hydrides)
    "H",
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def is_default_cation(symbol: str) -> bool:
    """
    Return True if the element is treated as a
    polyhedron center by default.
    """
    return symbol in DEFAULT_CATIONS


def is_default_anion(symbol: str) -> bool:
    """
    Return True if the element is treated as a
    ligand atom by default.
    """
    return symbol in DEFAULT_ANIONS
