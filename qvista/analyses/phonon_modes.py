"""
Phonon mode animation.

Turns an eigenvector into a sequence of displaced structures, so a mode can
be watched rather than read off a dispersion curve.

The physics is short: for a mode of frequency omega at wavevector q, the
displacement of atom *a* in the cell at lattice vector R is

    u_a(R, t) = Re[ (e_a / sqrt(m_a)) * exp(i (q.R - omega t)) ]

phonopy's eigenvectors are mass-weighted, which is why the ``1/sqrt(m)``
appears: without it, heavy atoms are drawn moving as far as light ones,
which is exactly backwards.

Animating the primitive cell means R = 0, so the frames sweep the phase of
``exp(-i omega t)`` over one period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: Frames in one oscillation period.
DEFAULT_FRAMES = 24

#: Largest atomic displacement in a frame, in Angstrom. Chosen to be
#: visible without atoms passing through one another.
DEFAULT_AMPLITUDE = 0.35


@dataclass
class PhononMode:
    """One normal mode at one wavevector."""

    label: str
    qpoint: tuple[float, float, float]

    band_index: int
    frequency: float

    #: Mass-weighted eigenvector, length 3N, complex.
    eigenvector: Any

    @property
    def is_imaginary(self) -> bool:
        """True for a soft mode, which phonopy reports as negative."""

        return self.frequency < 0.0

    def describe(self) -> str:
        """One-line summary for a list widget."""

        marker = " (imaginary)" if self.is_imaginary else ""

        return (
            f"Band {self.band_index + 1:>2}  "
            f"{self.frequency:8.3f} THz  @ {self.label}{marker}"
        )


def modes_at_qpoint(
    phonon,
    qpoint: Sequence[float],
    label: str = "",
) -> list[PhononMode]:
    """
    Return every normal mode at ``qpoint``.

    ``qpoint`` is fractional, in the reciprocal basis of the cell the
    force constants were built on.
    """

    import numpy as np

    phonon.run_qpoints([list(qpoint)], with_eigenvectors=True)

    result = phonon.qpoints

    frequencies = np.asarray(result.frequencies)[0]
    eigenvectors = np.asarray(result.eigenvectors)[0]

    return [
        PhononMode(
            label=label,
            qpoint=tuple(float(value) for value in qpoint),
            band_index=index,
            frequency=float(frequencies[index]),
            # Column index selects the mode.
            eigenvector=eigenvectors[:, index],
        )
        for index in range(len(frequencies))
    ]


def mode_displacements(
    eigenvector,
    masses: Sequence[float],
):
    """
    Convert a mass-weighted eigenvector into atomic displacement vectors.

    Returns a complex ``(natoms, 3)`` array. Taking the real part after
    multiplying by a phase gives the displacement at that instant.
    """

    import numpy as np

    vector = np.asarray(eigenvector).reshape(-1, 3)

    weights = np.asarray(masses, dtype=float) ** 0.5

    return vector / weights[:, None]


def animation_frames(
    positions,
    eigenvector,
    masses: Sequence[float],
    frames: int = DEFAULT_FRAMES,
    amplitude: float = DEFAULT_AMPLITUDE,
):
    """
    Generate one oscillation period as a list of Cartesian position arrays.

    ``positions`` are the equilibrium Cartesian coordinates. The
    displacement pattern is scaled so its largest atomic excursion equals
    ``amplitude``, which keeps every mode equally visible regardless of how
    the eigenvector happened to be normalised.
    """

    import numpy as np

    equilibrium = np.asarray(positions, dtype=float)

    pattern = mode_displacements(eigenvector, masses)

    largest = float(np.abs(pattern).max())

    if largest > 0.0:
        pattern = pattern / largest * amplitude

    out = []

    for index in range(max(1, int(frames))):

        phase = 2.0 * np.pi * index / max(1, int(frames))

        offset = np.real(pattern * np.exp(-1j * phase))

        out.append(equilibrium + offset)

    return out


def travelling_frames(
    cell,
    positions,
    masses: Sequence[float],
    eigenvector,
    qpoint: Sequence[float],
    repeat: Sequence[int] = (1, 1, 1),
    frames: int = DEFAULT_FRAMES,
    amplitude: float = DEFAULT_AMPLITUDE,
):
    """
    One period of a mode, across a repeated cell.

    :func:`animation_frames` animates a single cell, where ``R = 0`` and
    the phase factor drops out -- so a zone-boundary mode looks exactly
    like a mode at Gamma. Repeating the cell restores the part that
    makes a phonon a wave: the atom at lattice vector ``R`` lags the one
    at the origin by ``exp(i q.R)``, which for a fractional q-point and
    an integer cell index ``n`` is ``exp(2 pi i q.n)``.

    At ``q = 0`` every cell moves together and this reduces to the
    single-cell animation, repeated. Away from Gamma it is the
    difference between seeing a wave and seeing a rigid shuffle.

    Returns ``(frames, source, lattice, equilibrium)``: the position
    arrays, the index of the source atom for each repeated atom, the
    supercell lattice, and the undisplaced positions -- which are not
    ``frames[0]``, because frame zero is already a quarter of the way
    into the oscillation.
    """

    import numpy as np

    lattice = np.asarray(cell, dtype=float)

    base = np.asarray(positions, dtype=float)

    pattern = mode_displacements(eigenvector, masses)

    largest = float(np.abs(pattern).max())

    if largest > 0.0:
        pattern = pattern / largest * amplitude

    counts = [max(1, int(value)) for value in repeat]

    equilibrium = []
    displacement = []
    source = []

    for n0 in range(counts[0]):
        for n1 in range(counts[1]):
            for n2 in range(counts[2]):

                offset = np.array([n0, n1, n2], dtype=float) @ lattice

                phase = np.exp(
                    2.0j
                    * np.pi
                    * float(np.dot(np.asarray(qpoint, dtype=float),
                                   [n0, n1, n2]))
                )

                equilibrium.append(base + offset)
                displacement.append(pattern * phase)
                source.append(np.arange(len(base)))

    equilibrium = np.vstack(equilibrium)
    displacement = np.vstack(displacement)
    source = np.concatenate(source)

    total = max(1, int(frames))

    out = []

    for index in range(total):

        turn = np.exp(-2.0j * np.pi * index / total)

        out.append(equilibrium + np.real(displacement * turn))

    return (
        out,
        source,
        (lattice.T * np.asarray(counts, dtype=float)).T,
        equilibrium,
    )


def to_xyz_trajectory(
    symbols: Sequence[str],
    frames,
    cell=None,
    comment: str = "",
) -> str:
    """
    Render frames as a multi-frame extended XYZ trajectory.

    This is the portable way out of QVista: VESTA, OVITO and ASE's own
    viewer all animate this format directly.
    """

    import numpy as np

    lines: list[str] = []

    if cell is not None:
        flat = " ".join(
            f"{value:.8f}"
            for value in np.asarray(cell, dtype=float).reshape(-1)
        )
        lattice = f' Lattice="{flat}" Properties=species:S:1:pos:R:3'
    else:
        lattice = ""

    for index, positions in enumerate(frames):

        lines.append(str(len(symbols)))
        lines.append(f"{comment} frame={index + 1}{lattice}")

        for symbol, point in zip(symbols, np.asarray(positions)):
            lines.append(
                f"{symbol:<3} "
                f"{point[0]:14.8f} {point[1]:14.8f} {point[2]:14.8f}"
            )

    return "\n".join(lines) + "\n"
