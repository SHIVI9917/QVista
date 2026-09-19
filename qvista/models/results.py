"""
Container for analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Results:
    """Stores all computed analysis results."""

    ddec: dict[str, Any] = field(default_factory=dict)
    bader: dict[str, Any] = field(default_factory=dict)
    neb: dict[str, Any] = field(default_factory=dict)
    dos: dict[str, Any] = field(default_factory=dict)
    pdos: dict[str, Any] = field(default_factory=dict)
    bands: dict[str, Any] = field(default_factory=dict)

    #: Phonon band structure and force constants, from whichever route
    #: produced them last -- the machine-learning one or VASP.
    phonon: dict[str, Any] = field(default_factory=dict)

    #: The VASP phonon configuration last written, so postprocessing
    #: opens on the folder preprocessing created. Kept apart from
    #: :attr:`phonon` because writing input files must not evict force
    #: constants that have already been computed.
    phonon_vasp: dict[str, Any] = field(default_factory=dict)

    #: Phonon density of states.
    phonon_dos: dict[str, Any] = field(default_factory=dict)

    #: Elastic tensor and derived mechanical properties.
    elastic: dict[str, Any] = field(default_factory=dict)

    #: High-symmetry k-path through the Brillouin zone.
    kpath: dict[str, Any] = field(default_factory=dict)

    #: Space group, point group and Wyckoff positions.
    spacegroup: dict[str, Any] = field(default_factory=dict)

    #: Molecular dynamics time series.
    md: dict[str, Any] = field(default_factory=dict)

    #: The relaxed geometry and how it got there.
    relaxation: dict[str, Any] = field(default_factory=dict)
