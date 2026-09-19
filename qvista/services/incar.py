"""
INCAR generation.

An INCAR is a flat list of ``TAG = value`` lines telling VASP what kind of
calculation to run. Nothing in it is derived from the structure, so the
only thing standing between a user and a working input file is knowing
which two dozen tags out of several hundred matter for the job at hand,
and which values those tags accept.

The shape of this module follows the way the decision is actually made.
First comes the main job -- one of eight, from a geometry optimisation to
an optical spectrum. That choice brings its own tags and its own
defaults, because the cutoff that is generous for a relaxation is thin for
a phonon and the EDIFF that is fine for forces is useless for a Hessian.
Inside a job there may be a further either/or -- which ensemble for AIMD,
DFPT or finite differences for a phonon -- and each of those brings tags
of its own. A single point is the one job where the sub-choices are not
exclusive: a static SCF and a DOS and a Bader run are one calculation
with one INCAR, and each box ticked adds to it.

On top of any job sits optional physics: a Hubbard U, a hybrid, a
dispersion correction, spin-orbit, implicit solvation, a meta-GGA. None
of it is on until asked for. Underneath everything sit the tags that have
nothing to do with the science -- what to write to disk, how to split the
work across ranks, how to mix the charge density -- which are the same
whatever job is running.

The file is then assembled in that order, and a tag written twice keeps
the position of its first appearance and the value of its last. That rule
is what lets the global block have the final word on LWAVE without the
job block having to know the global block exists.

Nothing here imports Qt. The dialog reads :data:`JOBS` to lay itself out
and calls :func:`incar_text` to produce the file, which is also how the
tests drive it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from enum import StrEnum

from qvista.services.potcar import RELAX_ENCUT_FACTOR

#: Comment character. VASP stops reading a line at either ``!`` or ``#``.
COMMENT = "#"

#: VASP's own spelling of the logicals.
TRUE = ".TRUE."
FALSE = ".FALSE."


# ----------------------------------------------------------------
# ENCUT
# ----------------------------------------------------------------

#: Bounds for a derived cutoff, in eV. Below ~100 eV nothing converges;
#: above ~700 eV is past what any standard PAW potential asks for, and
#: the cost goes as ENCUT^(3/2).
ENCUT_MIN = 100
ENCUT_MAX = 700

#: Granularity of the ENCUT editors.
ENCUT_STEP = 10

#: What this module falls back to when no POTCAR library is set, and the
#: starting value every job carries.
ENCUT_FALLBACK = 520


def encut_for(enmax: float | None) -> int | None:
    """
    The cutoff to start from, given the largest ENMAX of the POTCARs.

    ``None`` when no ENMAX is known -- there is no sensible guess in that
    case, and picking one silently would be worse than saying so.

    VASP's advice for anything that changes the cell is 1.3 x the largest
    ENMAX, because a basis set fixed in a moving cell leaves Pulay stress
    in the result. The answer is clamped to the range above; a structure
    whose ENMAX is high enough to pass 700 eV is real (fluorine and the
    ``_h`` variants get there), and the clamp shows up to the caller as
    the returned value being :data:`ENCUT_MAX`.
    """

    if enmax is None:
        return None

    raised = math.ceil(enmax * RELAX_ENCUT_FACTOR)

    return max(ENCUT_MIN, min(ENCUT_MAX, raised))


# ----------------------------------------------------------------
# How a tag is edited
# ----------------------------------------------------------------


class Kind(StrEnum):
    """How one setting is presented."""

    #: Free text. EDIFF and friends, which are written ``1E-6``.
    TEXT = "text"

    #: A whole number.
    INT = "int"

    #: A number with a fractional part.
    FLOAT = "float"

    #: One of a fixed set of values, each with its meaning attached.
    CHOICE = "choice"

    #: On or off, written ``.TRUE.`` or ``.FALSE.``.
    BOOL = "bool"

    #: Shown, but not editable: the job only works with this value.
    FIXED = "fixed"


@dataclass(frozen=True)
class Choice:
    """One row of a dropdown: what gets written, and what it means."""

    #: Exactly what goes into the INCAR.
    value: str

    #: What the dropdown shows. Carrying the meaning beside the value is
    #: the difference between choosing a smearing scheme and remembering
    #: that ``ISMEAR = -5`` is the tetrahedron method.
    label: str


@dataclass(frozen=True)
class Field:
    """One editable setting."""

    #: Unique within the dialog. Also the key the builders read.
    name: str

    #: What the row is called -- the INCAR tag, in nearly every case.
    label: str

    kind: Kind

    #: Starting value, always as it would be written.
    default: str

    #: For :attr:`Kind.CHOICE`.
    choices: tuple[Choice, ...] = ()

    low: float | None = None
    high: float | None = None
    step: float = 1.0

    #: Digits after the point for :attr:`Kind.FLOAT`.
    decimals: int = 0

    #: Sits under the editor. Says what the tag does, not what it is.
    note: str = ""

    #: One editor per species in POSCAR order rather than one editor.
    #: LANGEVIN_GAMMA is the only tag here that works this way.
    per_species: bool = False


@dataclass(frozen=True)
class Section:
    """A titled run of settings."""

    title: str

    fields: tuple[Field, ...] = ()

    #: Conditions, each ``(field name, values that satisfy it)``. All of
    #: them have to hold; a section with none is always shown. This is
    #: how the ensemble choice reveals a thermostat and the phonon method
    #: reveals a displacement.
    #:
    #: More than one is needed because the thermostat sections belong to
    #: NVT specifically. Keyed on the thermostat alone they would follow
    #: it into NPT, where the setting they offer is not read.
    when: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def visible(self, values: dict[str, str]) -> bool:
        """Whether this section applies, given what has been chosen."""

        return all(
            values.get(name, "") in allowed for name, allowed in self.when
        )

    def gates(self) -> tuple[str, ...]:
        """The fields this section's visibility depends on."""

        return tuple(name for name, _ in self.when)


@dataclass(frozen=True)
class Job:
    """One of the eight main calculations."""

    name: str

    #: One line on the card saying what the job is for.
    tagline: str

    #: What SYSTEM starts as.
    system: str

    sections: tuple[Section, ...]

    def fields(self) -> tuple[Field, ...]:
        """Every field the job defines, shown or not."""

        return tuple(
            field for section in self.sections for field in section.fields
        )

    def defaults(self) -> dict[str, str]:
        """Every field at its starting value."""

        return {field.name: field.default for field in self.fields()}


# ----------------------------------------------------------------
# Values that recur across jobs
# ----------------------------------------------------------------

#: How the Brillouin zone is integrated. The tetrahedron method is exact
#: for a static run on a mesh and wrong on a k-path or for forces;
#: Methfessel-Paxton is the metal's choice and needs SIGMA checked
#: against the entropy term.
ISMEAR_CHOICES = (
    Choice("-5", "-5 - Tetrahedron with Blochl corrections"),
    Choice("-4", "-4 - Tetrahedron"),
    Choice("-3", "-3 - Loop over smearing parameters"),
    Choice("-2", "-2 - Occupancies read from input and kept fixed"),
    Choice("-1", "-1 - Fermi smearing"),
    Choice("0", "0 - Gaussian smearing"),
    Choice("1", "1 - Methfessel-Paxton, order 1"),
    Choice("2", "2 - Methfessel-Paxton, order 2"),
)

#: Whether the projection operators are evaluated in real space. Real
#: space is faster on large cells and costs a little accuracy, which is
#: why anything measuring a second derivative wants it off.
LREAL_AUTO_FIRST = (
    Choice("Auto", "Auto - chosen by VASP"),
    Choice(FALSE, "False - reciprocal space"),
    Choice(TRUE, "True - real space"),
)

LREAL_FALSE_FIRST = (
    Choice(FALSE, "False - reciprocal space"),
    Choice("Auto", "Auto - chosen by VASP"),
    Choice(TRUE, "True - real space"),
)

SPIN_CHOICES = (
    Choice("1", "1 - Non-spin-polarised"),
    Choice("2", "2 - Spin-polarised"),
)

ALGO_CHOICES = (
    Choice("Normal", "Normal - blocked Davidson"),
    Choice("Fast", "Fast - Davidson, then RMM-DIIS"),
    Choice("VeryFast", "VeryFast - RMM-DIIS"),
)

PREC_CHOICES = (
    Choice("Low", "Low"),
    Choice("Medium", "Medium"),
    Choice("Normal", "Normal"),
    Choice("High", "High"),
    Choice("Accurate", "Accurate - no aliasing errors in the forces"),
)

LOGICAL_CHOICES = (Choice(TRUE, "TRUE"), Choice(FALSE, "FALSE"))

def _logical(name: str, label: str, default: str, note: str = "") -> Field:
    """A tag that is either ``.TRUE.`` or ``.FALSE.``, as a dropdown."""

    return Field(
        name,
        label,
        Kind.CHOICE,
        default,
        choices=LOGICAL_CHOICES,
        note=note,
    )


def _encut(name: str) -> Field:

    return Field(
        name,
        "ENCUT",
        Kind.INT,
        str(ENCUT_FALLBACK),
        low=1,
        step=ENCUT_STEP,
        note="Plane-wave cutoff in eV.",
    )


def _ediff(name: str, default: str) -> Field:

    return Field(
        name,
        "EDIFF",
        Kind.TEXT,
        default,
        note="Electronic convergence, in eV.",
    )


def _nelm(name: str, default: str = "100") -> Field:

    return Field(
        name,
        "NELM",
        Kind.INT,
        default,
        low=1,
        step=10,
        note="Most electronic steps allowed in one ionic step.",
    )


def _nelmin(name: str, default: str) -> Field:

    return Field(
        name,
        "NELMIN",
        Kind.INT,
        default,
        low=1,
        note="Fewest electronic steps before the ionic step is taken.",
    )


def _ismear(name: str, default: str) -> Field:

    return Field(
        name,
        "ISMEAR",
        Kind.CHOICE,
        default,
        choices=ISMEAR_CHOICES,
        note="How partial occupancies are set.",
    )


def _sigma(name: str) -> Field:

    return Field(
        name,
        "SIGMA",
        Kind.FLOAT,
        "0.05",
        low=0.001,
        step=0.01,
        decimals=3,
        note="Width of the smearing, in eV.",
    )


def _ispin(name: str) -> Field:

    return Field(
        name,
        "ISPIN",
        Kind.CHOICE,
        "2",
        choices=SPIN_CHOICES,
        note="Whether the two spin channels are treated separately.",
    )


def _algo(name: str) -> Field:

    return Field(
        name,
        "ALGO",
        Kind.CHOICE,
        "Normal",
        choices=ALGO_CHOICES,
        note="Which electronic minimisation is used.",
    )


def _prec(name: str, default: str) -> Field:

    return Field(
        name,
        "PREC",
        Kind.CHOICE,
        default,
        choices=PREC_CHOICES,
        note="Sets the FFT grids and the real-space projection radii.",
    )


def _lreal(name: str, choices, default: str, note: str = "") -> Field:

    return Field(
        name,
        "LREAL",
        Kind.CHOICE,
        default,
        choices=choices,
        note=note or "Where the projection operators are evaluated.",
    )


def _system(name: str, default: str) -> Field:

    return Field(
        name,
        "SYSTEM",
        Kind.TEXT,
        default,
        note="A name for the run. VASP does nothing with it.",
    )


def _potim_displacement(name: str) -> Field:

    return Field(
        name,
        "POTIM",
        Kind.FLOAT,
        "0.015",
        low=0.001,
        step=0.001,
        decimals=3,
        note="Displacement amplitude for the finite differences, in A.",
    )


# ----------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------

ISIF_CHOICES = (
    Choice("0", "0 - Forces only"),
    Choice("1", "1 - Forces and the trace of the stress tensor"),
    Choice("2", "2 - Ions move, cell fixed"),
    Choice("3", "3 - Ions, cell shape and cell volume all relax"),
    Choice("4", "4 - Ions and cell shape, volume fixed"),
    Choice("5", "5 - Cell shape only"),
    Choice("6", "6 - Cell shape and volume, ions fixed"),
    Choice("7", "7 - Volume only"),
    Choice("8", "8 - Cell shape fixed"),
)

OPTIMIZATION = Job(
    name="Optimization",
    tagline="Relax the ions, the cell, or both.",
    system="VASP Optimization",
    sections=(
        Section(
            "Optimization INCAR Parameters",
            (
                _system("optimization_system", "VASP Optimization"),
                _encut("optimization_encut"),
                Field(
                    "optimization_ibrion",
                    "IBRION",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("1", "1 - Quasi-Newton"),
                        Choice("2", "2 - Conjugate gradient"),
                        Choice("3", "3 - Damped molecular dynamics"),
                    ),
                    note="How the ions are moved downhill.",
                ),
                Field(
                    "optimization_isif",
                    "ISIF",
                    Kind.CHOICE,
                    "2",
                    choices=ISIF_CHOICES,
                    note="What is allowed to relax.",
                ),
                Field(
                    "optimization_nsw",
                    "NSW",
                    Kind.INT,
                    "200",
                    low=0,
                    step=10,
                    note="Most ionic steps allowed.",
                ),
                _ediff("optimization_ediff", "1E-5"),
                Field(
                    "optimization_ediffg",
                    "EDIFFG",
                    Kind.TEXT,
                    "-0.02",
                    note=(
                        "Ionic convergence. Negative is a force threshold "
                        "in eV/A; positive is an energy threshold in eV."
                    ),
                ),
                _nelm("optimization_nelm"),
                _nelmin("optimization_nelmin", "2"),
                _ismear("optimization_ismear", "0"),
                _sigma("optimization_sigma"),
                _ispin("optimization_ispin"),
                _lreal(
                    "optimization_lreal", LREAL_AUTO_FIRST, "Auto"
                ),
            ),
        ),
    ),
)


def _build_optimization(setup: Setup) -> dict[str, str]:

    values = setup.values

    return {
        "SYSTEM": values["optimization_system"],
        "ENCUT": values["optimization_encut"],
        "IBRION": values["optimization_ibrion"],
        "ISIF": values["optimization_isif"],
        "NSW": values["optimization_nsw"],
        "EDIFF": values["optimization_ediff"],
        "EDIFFG": values["optimization_ediffg"],
        "NELM": values["optimization_nelm"],
        "NELMIN": values["optimization_nelmin"],
        "ISMEAR": values["optimization_ismear"],
        "SIGMA": values["optimization_sigma"],
        "ISPIN": values["optimization_ispin"],
        "LREAL": values["optimization_lreal"],
    }


# ----------------------------------------------------------------
# Single-Point
# ----------------------------------------------------------------

#: The things a single self-consistent run can be asked to produce. Any
#: number of them at once: they are one calculation, and ticking a second
#: one adds tags rather than starting again.
SUBJOBS: tuple[str, ...] = (
    "Static SCF",
    "Band Structure",
    "DOS",
    "PDOS",
    "Work Function",
    "Bader Charge",
    "COHP / ICOHP",
    "D-Band Center",
    "P-Band Center",
    "Charge Density",
    "ELF",
    "Potential / LOCPOT",
)

#: What each sub-job is asking VASP for, shown beside its box.
SUBJOB_NOTES: dict[str, str] = {
    "Static SCF": "One self-consistent cycle at fixed geometry.",
    "Band Structure": "Eigenvalues along a k-path.",
    "DOS": "Density of states on a dense mesh.",
    "PDOS": "Site- and orbital-projected density of states.",
    "Work Function": "Planar-averaged potential in the vacuum.",
    "Bader Charge": "All-electron density for a charge partition.",
    "COHP / ICOHP": "Projected wavefunctions for a bonding analysis.",
    "D-Band Center": "Projected d states, for the first moment.",
    "P-Band Center": "Projected p states, for the first moment.",
    "Charge Density": "CHGCAR for plotting or a restart.",
    "ELF": "Electron localisation function.",
    "Potential / LOCPOT": "Local potential on the FFT grid.",
}

#: Sub-jobs that need the projections written out.
PROJECTED = (
    "Band Structure",
    "DOS",
    "PDOS",
    "D-Band Center",
    "P-Band Center",
    "COHP / ICOHP",
)

#: Bands per atom, by block. A rough count of the valence orbitals a PAW
#: potential brings: one s, three more for the p shell, five more for
#: the d shell. It is an estimate and the editor says so, but it is much
#: closer than VASP's own default for anything that needs empty states.
S_BLOCK = frozenset(
    {
        "H", "He",
        "Li", "Be",
        "Na", "Mg",
        "K", "Ca",
        "Rb", "Sr",
        "Cs", "Ba",
        "Fr", "Ra",
    }
)

P_BLOCK = frozenset(
    {
        "B", "C", "N", "O", "F", "Ne",
        "Al", "Si", "P", "S", "Cl", "Ar",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
        "In", "Sn", "Sb", "Te", "I", "Xe",
        "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    }
)

D_BLOCK = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    }
)

BANDS_PER_ATOM = (
    (S_BLOCK, 1, "s-block"),
    (P_BLOCK, 4, "p-block"),
    (D_BLOCK, 9, "d-block"),
)

#: A bonding analysis projects onto a local basis and needs empty states
#: well above the Fermi level to do it, so the band count is doubled.
COHP_BAND_FACTOR = 2


def bands_for(symbol: str) -> tuple[int, str]:
    """
    Bands one atom of this element contributes, and which block it is in.

    ``(0, "unknown")`` for anything not classified -- the lanthanides and
    actinides, whose f electrons make any single number wrong. The caller
    shows the zero rather than guessing.
    """

    for block, bands, name in BANDS_PER_ATOM:

        if symbol in block:
            return bands, name

    return 0, "unknown"


LORBIT_CHOICES = (
    Choice("0", "0 - No projections"),
    Choice("1", "1 - Projections, RWIGS radii"),
    Choice("2", "2 - Projections with phase, RWIGS radii"),
    Choice("5", "5 - Projections with phase"),
    Choice("10", "10 - Projections, no RWIGS needed"),
    Choice("11", "11 - Projections resolved by l and m"),
    Choice("12", "12 - Projections by l and m, with phase"),
)

ICHARG_CHOICES = (
    Choice("0", "0 - From the initial orbitals"),
    Choice("1", "1 - Read CHGCAR"),
    Choice("2", "2 - Superposition of atomic densities"),
    Choice("4", "4 - Read the potential from POT"),
    Choice("10", "10 - Non-self-consistent, from the orbitals"),
    Choice("11", "11 - Non-self-consistent, read CHGCAR"),
    Choice("12", "12 - Non-self-consistent, atomic densities"),
)

SINGLE_POINT = Job(
    name="Single-Point",
    tagline="One SCF, and whatever is to be read off it.",
    system="VASP Single-Point",
    sections=(
        Section(
            "Common Single-Point Parameters",
            (
                _system("sp_system", "VASP Single-Point"),
                _encut("sp_encut"),
                _ediff("sp_ediff", "1E-6"),
                _nelm("sp_nelm"),
                _ismear("sp_ismear", "-5"),
                _sigma("sp_sigma"),
                _ispin("sp_ispin"),
                _lreal("sp_lreal", LREAL_AUTO_FIRST, "Auto"),
                _algo("sp_algo"),
            ),
        ),
        Section(
            "Electronic Structure Parameters",
            (
                Field(
                    "sp_nedos",
                    "NEDOS",
                    Kind.INT,
                    "2000",
                    low=100,
                    step=100,
                    note="Points on the energy grid of the DOS.",
                ),
                Field(
                    "sp_lorbit",
                    "LORBIT",
                    Kind.CHOICE,
                    "11",
                    choices=LORBIT_CHOICES,
                    note="How much projection detail is written.",
                ),
            ),
        ),
        Section(
            "Work Function Parameters",
            (
                Field(
                    "sp_lvhar",
                    "LVHAR",
                    Kind.BOOL,
                    TRUE,
                    note="Write the ionic and Hartree potential only.",
                ),
                Field(
                    "sp_lvtot",
                    "LVTOT",
                    Kind.BOOL,
                    TRUE,
                    note="Write the full local potential to LOCPOT.",
                ),
            ),
        ),
        Section(
            "Bader Charge Parameters",
            (
                Field(
                    "sp_laechg",
                    "LAECHG",
                    Kind.BOOL,
                    TRUE,
                    note=(
                        "Write the core and all-electron valence density. "
                        "A Bader partition on the pseudo-density alone is "
                        "wrong."
                    ),
                ),
            ),
        ),
        Section(
            "COHP / ICOHP Parameters",
            (
                Field(
                    "sp_cohp_lorbit",
                    "LORBIT",
                    Kind.CHOICE,
                    "11",
                    choices=(
                        Choice("10", "10 - Projections, no RWIGS needed"),
                        Choice("11", "11 - Projections resolved by l and m"),
                        Choice("12", "12 - Projections by l and m, with phase"),
                    ),
                    note="LOBSTER reads the l- and m-resolved projections.",
                ),
                Field(
                    "sp_cohp_lwave",
                    "LWAVE",
                    Kind.BOOL,
                    TRUE,
                    note="LOBSTER needs the WAVECAR.",
                ),
            ),
        ),
        Section(
            "Charge Density Parameters",
            (
                Field(
                    "sp_icharg",
                    "ICHARG",
                    Kind.CHOICE,
                    "0",
                    choices=ICHARG_CHOICES,
                    note="Where the starting charge density comes from.",
                ),
            ),
        ),
        Section(
            "ELF Parameters",
            (
                Field(
                    "sp_lelf",
                    "LELF",
                    Kind.BOOL,
                    TRUE,
                    note="Write the electron localisation function.",
                ),
            ),
        ),
        Section(
            "Potential / LOCPOT Parameters",
            (
                Field(
                    "sp_potential_lvhar",
                    "LVHAR",
                    Kind.BOOL,
                    TRUE,
                    note="Ionic and Hartree potential only.",
                ),
                Field(
                    "sp_potential_lvtot",
                    "LVTOT",
                    Kind.BOOL,
                    TRUE,
                    note="The full local potential, exchange included.",
                ),
            ),
        ),
    ),
)

#: Which section of the single-point job each sub-job brings with it.
#: The electronic-structure section is shared by six of them, which is
#: why it is keyed by a tuple.
SUBJOB_SECTIONS: dict[str, str] = {
    "Work Function": "Work Function Parameters",
    "Bader Charge": "Bader Charge Parameters",
    "COHP / ICOHP": "COHP / ICOHP Parameters",
    "Charge Density": "Charge Density Parameters",
    "ELF": "ELF Parameters",
    "Potential / LOCPOT": "Potential / LOCPOT Parameters",
}


def single_point_sections(subjobs) -> tuple[str, ...]:
    """
    Which sections of the single-point job apply, given the ticked boxes.

    Always the common parameters. The electronic-structure block as soon
    as anything wants projections, and then one block per sub-job that
    has settings of its own.
    """

    chosen = set(subjobs)

    names = ["Common Single-Point Parameters"]

    if chosen & set(PROJECTED):
        names.append("Electronic Structure Parameters")

    for subjob in SUBJOBS:

        section = SUBJOB_SECTIONS.get(subjob)

        if section is not None and subjob in chosen:
            names.append(section)

    return tuple(names)


def _build_single_point(setup: Setup) -> dict[str, str]:

    values = setup.values
    chosen = set(setup.subjobs)

    parameters = {
        "SYSTEM": values["sp_system"],
        "ENCUT": values["sp_encut"],
        "EDIFF": values["sp_ediff"],
        "NELM": values["sp_nelm"],
        "ISMEAR": values["sp_ismear"],
        "SIGMA": values["sp_sigma"],
        "ISPIN": values["sp_ispin"],
        # A single point is a single point: no ionic motion, whatever
        # else has been asked for.
        "IBRION": "-1",
        "NSW": "0",
        "LREAL": values["sp_lreal"],
        "ALGO": values["sp_algo"],
    }

    bands = setup.nbands(doubled="COHP / ICOHP" in chosen)

    if bands is not None:
        parameters["NBANDS"] = str(bands)

    if chosen & set(PROJECTED):
        parameters["NEDOS"] = values["sp_nedos"]
        parameters["LORBIT"] = values["sp_lorbit"]

    if "Work Function" in chosen:
        parameters["LVHAR"] = values["sp_lvhar"]
        parameters["LVTOT"] = values["sp_lvtot"]

    if "Bader Charge" in chosen:
        parameters["LAECHG"] = values["sp_laechg"]
        parameters["LCHARG"] = TRUE

    if "COHP / ICOHP" in chosen:
        # LOBSTER is stricter about the projections than a plain PDOS,
        # so where both are asked for this is the one that stands.
        parameters["LORBIT"] = values["sp_cohp_lorbit"]
        parameters["LWAVE"] = values["sp_cohp_lwave"]

    if "Charge Density" in chosen:
        parameters["ICHARG"] = values["sp_icharg"]
        parameters["LCHARG"] = TRUE

    if "ELF" in chosen:
        parameters["LELF"] = values["sp_lelf"]

    if "Potential / LOCPOT" in chosen:
        parameters["LVHAR"] = values["sp_potential_lvhar"]
        parameters["LVTOT"] = values["sp_potential_lvtot"]

    return parameters


# ----------------------------------------------------------------
# AIMD
# ----------------------------------------------------------------

ENSEMBLES = ("NVT", "NVE", "NPT")

#: Thermostat name to MDALGO. The name is what a user recognises; the
#: number is what VASP reads.
THERMOSTATS: dict[str, str] = {
    "Nose-Hoover": "2",
    "Andersen": "1",
    "Langevin": "3",
    "Nose-Hoover Chain": "4",
    "CSVR": "5",
}

AIMD = Job(
    name="AIMD",
    tagline="Born-Oppenheimer dynamics in NVT, NVE or NPT.",
    system="VASP AIMD",
    sections=(
        Section(
            "MD Ensemble",
            (
                Field(
                    "aimd_ensemble",
                    "Ensemble",
                    Kind.CHOICE,
                    "NVT",
                    choices=(
                        Choice("NVT", "NVT - fixed cell, thermostatted"),
                        Choice("NVE", "NVE - fixed cell, constant energy"),
                        Choice("NPT", "NPT - variable cell and temperature"),
                    ),
                    note="Which quantities are held fixed.",
                ),
            ),
        ),
        Section(
            "General MD Parameters",
            (
                _system("aimd_system", "VASP AIMD"),
                _encut("aimd_encut"),
                _ediff("aimd_ediff", "1E-5"),
                Field(
                    "aimd_nsw",
                    "NSW",
                    Kind.INT,
                    "10000",
                    low=1,
                    step=100,
                    note="Number of MD steps.",
                ),
                Field(
                    "aimd_potim",
                    "POTIM",
                    Kind.FLOAT,
                    "1.0",
                    low=0.01,
                    step=0.1,
                    decimals=2,
                    note="Time step, in fs.",
                ),
                Field(
                    "aimd_isym",
                    "ISYM",
                    Kind.CHOICE,
                    "0",
                    choices=(
                        Choice("0", "0 - Symmetry off"),
                        Choice(
                            "-1",
                            "-1 - Symmetry off, no symmetrisation at all",
                        ),
                    ),
                    note=(
                        "Symmetry has to be off: the trajectory breaks it "
                        "on the first step."
                    ),
                ),
            ),
        ),
        Section(
            "Electronic Parameters",
            (
                _nelm("aimd_nelm"),
                _nelmin("aimd_nelmin", "4"),
                _ispin("aimd_ispin"),
                _ismear("aimd_ismear", "-5"),
                _sigma("aimd_sigma"),
                _algo("aimd_algo"),
            ),
        ),
        Section(
            "MD Settings",
            (
                Field(
                    "aimd_ibrion",
                    "IBRION",
                    Kind.FIXED,
                    "0",
                    note="IBRION = 0 is molecular dynamics.",
                ),
                _lreal("aimd_lreal", LREAL_AUTO_FIRST, "Auto"),
            ),
        ),
        # ------------------------------------------------------
        # NVT
        # ------------------------------------------------------
        Section(
            "Temperature",
            (
                Field(
                    "aimd_nvt_tebeg",
                    "TEBEG",
                    Kind.FLOAT,
                    "300.0",
                    low=0.1,
                    step=10.0,
                    decimals=1,
                    note="Starting temperature, in K.",
                ),
                Field(
                    "aimd_nvt_teend",
                    "TEEND",
                    Kind.FLOAT,
                    "300.0",
                    low=0.1,
                    step=10.0,
                    decimals=1,
                    note="Final temperature. Equal to TEBEG is isothermal.",
                ),
                Field(
                    "aimd_nvt_isif",
                    "ISIF",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("0", "0 - Forces only"),
                        Choice("1", "1 - Forces and the trace of the stress"),
                        Choice("2", "2 - Forces and the full stress tensor"),
                    ),
                    note="Below 3 the cell stays fixed, which NVT requires.",
                ),
            ),
            when=(("aimd_ensemble", ("NVT",)),),
        ),
        Section(
            "Thermostat",
            (
                Field(
                    "aimd_nvt_thermostat",
                    "Thermostat",
                    Kind.CHOICE,
                    "Nose-Hoover",
                    choices=(
                        Choice("Nose-Hoover", "Nose-Hoover (MDALGO = 2)"),
                        Choice("Andersen", "Andersen (MDALGO = 1)"),
                        Choice("Langevin", "Langevin (MDALGO = 3)"),
                        Choice(
                            "Nose-Hoover Chain",
                            "Nose-Hoover Chain (MDALGO = 4)",
                        ),
                        Choice("CSVR", "CSVR (MDALGO = 5)"),
                    ),
                    note="How the heat bath is coupled to the ions.",
                ),
            ),
            when=(("aimd_ensemble", ("NVT",)),),
        ),
        Section(
            "Nose-Hoover",
            (
                Field(
                    "aimd_nvt_smass",
                    "SMASS",
                    Kind.FLOAT,
                    "0.0",
                    low=0.0,
                    step=0.1,
                    decimals=2,
                    note=(
                        "Mass of the thermostat. Too small and the "
                        "temperature rings instead of settling."
                    ),
                ),
            ),
            when=(
                ("aimd_ensemble", ("NVT",)),
                ("aimd_nvt_thermostat", ("Nose-Hoover",)),
            ),
        ),
        Section(
            "Andersen",
            (
                Field(
                    "aimd_nvt_andersen_prob",
                    "ANDERSEN_PROB",
                    Kind.FLOAT,
                    "0.1",
                    low=0.0,
                    high=1.0,
                    step=0.01,
                    decimals=3,
                    note="Chance per step that a velocity is redrawn.",
                ),
            ),
            when=(
                ("aimd_ensemble", ("NVT",)),
                ("aimd_nvt_thermostat", ("Andersen",)),
            ),
        ),
        Section(
            "LANGEVIN_GAMMA",
            (
                Field(
                    "aimd_nvt_gamma",
                    "LANGEVIN_GAMMA",
                    Kind.FLOAT,
                    "10.0",
                    low=0.0,
                    step=1.0,
                    decimals=2,
                    per_species=True,
                    note="Friction coefficient, in ps^-1, one per species.",
                ),
            ),
            when=(
                ("aimd_ensemble", ("NVT",)),
                ("aimd_nvt_thermostat", ("Langevin",)),
            ),
        ),
        Section(
            "Nose-Hoover Chain",
            (
                Field(
                    "aimd_nvt_nhc_nchains",
                    "NHC_NCHAINS",
                    Kind.INT,
                    "3",
                    low=1,
                    note="Thermostats in the chain.",
                ),
                Field(
                    "aimd_nvt_nhc_period",
                    "NHC_PERIOD",
                    Kind.INT,
                    "3",
                    low=1,
                    note="Steps between chain updates.",
                ),
            ),
            when=(
                ("aimd_ensemble", ("NVT",)),
                ("aimd_nvt_thermostat", ("Nose-Hoover Chain",)),
            ),
        ),
        Section(
            "CSVR",
            (
                Field(
                    "aimd_nvt_csvr_period",
                    "CSVR_PERIOD",
                    Kind.INT,
                    "20",
                    low=1,
                    note="Steps between velocity rescalings.",
                ),
            ),
            when=(
                ("aimd_ensemble", ("NVT",)),
                ("aimd_nvt_thermostat", ("CSVR",)),
            ),
        ),
        # ------------------------------------------------------
        # NVE
        # ------------------------------------------------------
        Section(
            "Initial Temperature",
            (
                Field(
                    "aimd_nve_tebeg",
                    "TEBEG",
                    Kind.FLOAT,
                    "300.0",
                    low=0.1,
                    step=10.0,
                    decimals=1,
                    note=(
                        "Only used to draw the starting velocities, and "
                        "only if the POSCAR does not carry them."
                    ),
                ),
                Field(
                    "aimd_nve_isif",
                    "ISIF",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("0", "0 - Forces only"),
                        Choice("1", "1 - Forces and the trace of the stress"),
                        Choice("2", "2 - Forces and the full stress tensor"),
                    ),
                    note="The cell is fixed in NVE.",
                ),
            ),
            when=(("aimd_ensemble", ("NVE",)),),
        ),
        Section(
            "NVE Propagation",
            (
                Field(
                    "aimd_nve_method",
                    "Method",
                    Kind.CHOICE,
                    "Andersen propagation",
                    choices=(
                        Choice(
                            "Andersen propagation",
                            "Andersen with the collision rate at zero",
                        ),
                        Choice(
                            "Nose-Hoover propagation",
                            "Nose-Hoover with the thermostat switched off",
                        ),
                    ),
                    note=(
                        "Two ways to reach the same thing: a thermostatted "
                        "integrator with its coupling turned off."
                    ),
                ),
            ),
            when=(("aimd_ensemble", ("NVE",)),),
        ),
        # ------------------------------------------------------
        # NPT
        # ------------------------------------------------------
        Section(
            "Thermostat Temperature",
            (
                Field(
                    "aimd_npt_tebeg",
                    "TEBEG",
                    Kind.FLOAT,
                    "300.0",
                    low=0.1,
                    step=10.0,
                    decimals=1,
                    note="Starting temperature, in K.",
                ),
                Field(
                    "aimd_npt_teend",
                    "TEEND",
                    Kind.FLOAT,
                    "300.0",
                    low=0.1,
                    step=10.0,
                    decimals=1,
                    note="Final temperature, in K.",
                ),
            ),
            when=(("aimd_ensemble", ("NPT",)),),
        ),
        Section(
            "NPT Dynamics",
            (
                Field(
                    "aimd_npt_mdalgo",
                    "MDALGO",
                    Kind.FIXED,
                    "3",
                    note="NPT in VASP is the Langevin integrator.",
                ),
                Field(
                    "aimd_npt_isif",
                    "ISIF",
                    Kind.FIXED,
                    "3",
                    note="The cell has to be free for the barostat to act.",
                ),
            ),
            when=(("aimd_ensemble", ("NPT",)),),
        ),
        Section(
            "Ionic Langevin Thermostat",
            (
                Field(
                    "aimd_npt_gamma",
                    "LANGEVIN_GAMMA",
                    Kind.FLOAT,
                    "10.0",
                    low=0.0,
                    step=1.0,
                    decimals=2,
                    per_species=True,
                    note=(
                        "One friction coefficient per species, in ps^-1, "
                        "in POSCAR order."
                    ),
                ),
            ),
            when=(("aimd_ensemble", ("NPT",)),),
        ),
        Section(
            "Parrinello-Rahman Barostat",
            (
                Field(
                    "aimd_npt_gamma_l",
                    "LANGEVIN_GAMMA_L",
                    Kind.FLOAT,
                    "3.0",
                    low=0.01,
                    step=0.5,
                    decimals=2,
                    note="Friction on the lattice degrees of freedom, ps^-1.",
                ),
                Field(
                    "aimd_npt_pmass",
                    "PMASS",
                    Kind.FLOAT,
                    "1000.0",
                    low=1.0,
                    step=50.0,
                    decimals=1,
                    note="Fictitious mass of the lattice, in amu.",
                ),
                Field(
                    "aimd_npt_pstress",
                    "PSTRESS",
                    Kind.FLOAT,
                    "0.0",
                    step=0.1,
                    decimals=3,
                    note="Target external pressure, in kBar.",
                ),
            ),
            when=(("aimd_ensemble", ("NPT",)),),
        ),
        # ------------------------------------------------------
        Section(
            "AIMD Output",
            (
                _logical(
                    "aimd_lwave",
                    "LWAVE",
                    FALSE,
                    note="A WAVECAR per run is rarely worth the disk.",
                ),
                _logical(
                    "aimd_lcharg",
                    "LCHARG",
                    FALSE,
                    note="A CHGCAR per step is rarely worth the disk either.",
                ),
                Field(
                    "aimd_nblock",
                    "NBLOCK",
                    Kind.INT,
                    "1",
                    low=1,
                    note="MD steps between XDATCAR frames.",
                ),
            ),
        ),
    ),
)


def _build_aimd(setup: Setup) -> dict[str, str]:

    values = setup.values
    ensemble = values["aimd_ensemble"]

    parameters = {
        "SYSTEM": values["aimd_system"],
        "ENCUT": values["aimd_encut"],
        "EDIFF": values["aimd_ediff"],
        "IBRION": "0",
        "NSW": values["aimd_nsw"],
        "POTIM": values["aimd_potim"],
        "NELM": values["aimd_nelm"],
        "NELMIN": values["aimd_nelmin"],
        "ISMEAR": values["aimd_ismear"],
        "SIGMA": values["aimd_sigma"],
        "ISPIN": values["aimd_ispin"],
        "ISYM": values["aimd_isym"],
        "ALGO": values["aimd_algo"],
        "LWAVE": values["aimd_lwave"],
        "LCHARG": values["aimd_lcharg"],
        "NBLOCK": values["aimd_nblock"],
        "LREAL": values["aimd_lreal"],
    }

    if ensemble == "NVT":

        parameters["TEBEG"] = values["aimd_nvt_tebeg"]
        parameters["TEEND"] = values["aimd_nvt_teend"]

        thermostat = values["aimd_nvt_thermostat"]
        mdalgo = THERMOSTATS[thermostat]

        parameters["MDALGO"] = mdalgo
        parameters["ISIF"] = values["aimd_nvt_isif"]

        if thermostat == "Andersen":
            parameters["ANDERSEN_PROB"] = values["aimd_nvt_andersen_prob"]

        elif thermostat == "Nose-Hoover":
            parameters["SMASS"] = values["aimd_nvt_smass"]

        elif thermostat == "Langevin":

            gamma = setup.per_species("aimd_nvt_gamma")

            if gamma:
                parameters["LANGEVIN_GAMMA"] = gamma

        elif thermostat == "Nose-Hoover Chain":
            parameters["NHC_NCHAINS"] = values["aimd_nvt_nhc_nchains"]
            parameters["NHC_PERIOD"] = values["aimd_nvt_nhc_period"]

        elif thermostat == "CSVR":
            parameters["CSVR_PERIOD"] = values["aimd_nvt_csvr_period"]

    elif ensemble == "NVE":

        parameters["TEBEG"] = values["aimd_nve_tebeg"]
        parameters["ISIF"] = values["aimd_nve_isif"]

        if values["aimd_nve_method"] == "Andersen propagation":
            parameters["MDALGO"] = "1"
            parameters["ANDERSEN_PROB"] = "0.0"

        else:
            parameters["MDALGO"] = "2"
            parameters["SMASS"] = "-3"

    elif ensemble == "NPT":

        parameters["TEBEG"] = values["aimd_npt_tebeg"]
        parameters["TEEND"] = values["aimd_npt_teend"]
        parameters["MDALGO"] = "3"
        parameters["ISIF"] = "3"

        gamma = setup.per_species("aimd_npt_gamma")

        if gamma:
            parameters["LANGEVIN_GAMMA"] = gamma

        parameters["LANGEVIN_GAMMA_L"] = values["aimd_npt_gamma_l"]
        parameters["PMASS"] = values["aimd_npt_pmass"]
        parameters["PSTRESS"] = values["aimd_npt_pstress"]

    return parameters


# ----------------------------------------------------------------
# NEB
# ----------------------------------------------------------------

NEB = Job(
    name="NEB",
    tagline="Find the barrier between two minima.",
    system="VASP NEB",
    sections=(
        Section(
            "Nudged Elastic Band",
            (_system("neb_system", "VASP NEB"),),
        ),
        Section(
            "NEB Path",
            (
                Field(
                    "neb_images",
                    "IMAGES",
                    Kind.INT,
                    "5",
                    low=1,
                    note=(
                        "Intermediate images only. The two endpoints are "
                        "not counted and are not relaxed."
                    ),
                ),
                Field(
                    "neb_spring",
                    "SPRING",
                    Kind.FLOAT,
                    "-5.0",
                    step=0.5,
                    decimals=2,
                    note=(
                        "Spring constant between images. -5 is the tangent "
                        "projection, which is what makes it a nudged band."
                    ),
                ),
                _logical(
                    "neb_lclimb",
                    "LCLIMB",
                    FALSE,
                    note=(
                        "Let the highest image climb to the saddle. Worth "
                        "switching on once the band has roughly converged."
                    ),
                ),
            ),
        ),
        Section(
            "NEB Ionic Parameters",
            (
                Field(
                    "neb_ibrion",
                    "IBRION",
                    Kind.FIXED,
                    "3",
                    note="Damped dynamics, the only mover NEB works with.",
                ),
                Field(
                    "neb_potim",
                    "POTIM",
                    Kind.FIXED,
                    "0",
                    note="Zero hands the step size to the optimiser.",
                ),
                Field(
                    "neb_isif",
                    "ISIF",
                    Kind.FIXED,
                    "2",
                    note="The cell is common to every image and stays fixed.",
                ),
                Field(
                    "neb_nsw",
                    "NSW",
                    Kind.INT,
                    "200",
                    low=1,
                    step=10,
                    note="Most ionic steps allowed.",
                ),
                Field(
                    "neb_ediffg",
                    "EDIFFG",
                    Kind.TEXT,
                    "-0.02",
                    note=(
                        "Negative is a force threshold in eV/A, which is "
                        "the only sensible criterion for a band."
                    ),
                ),
            ),
        ),
        Section(
            "Electronic Parameters",
            (
                _encut("neb_encut"),
                _prec("neb_prec", "Normal"),
                _ediff("neb_ediff", "1E-5"),
                _nelm("neb_nelm"),
                _nelmin("neb_nelmin", "2"),
                _ismear("neb_ismear", "0"),
                _sigma("neb_sigma"),
                _ispin("neb_ispin"),
                Field(
                    "neb_isym",
                    "ISYM",
                    Kind.CHOICE,
                    "0",
                    choices=(
                        Choice("0", "0 - Symmetry off"),
                        Choice("-1", "-1 - Symmetry off, no symmetrisation"),
                    ),
                    note=(
                        "The images along the path do not share the "
                        "symmetry of the endpoints."
                    ),
                ),
                _algo("neb_algo"),
                _lreal("neb_lreal", LREAL_AUTO_FIRST, "Auto"),
            ),
        ),
    ),
)


def _build_neb(setup: Setup) -> dict[str, str]:

    values = setup.values

    return {
        "SYSTEM": values["neb_system"],
        "ENCUT": values["neb_encut"],
        "PREC": values["neb_prec"],
        "IBRION": "3",
        "POTIM": "0",
        "ISIF": "2",
        "IMAGES": values["neb_images"],
        "SPRING": values["neb_spring"],
        "LCLIMB": values["neb_lclimb"],
        "NSW": values["neb_nsw"],
        "EDIFF": values["neb_ediff"],
        "EDIFFG": values["neb_ediffg"],
        "NELM": values["neb_nelm"],
        "NELMIN": values["neb_nelmin"],
        "ISMEAR": values["neb_ismear"],
        "SIGMA": values["neb_sigma"],
        "ISPIN": values["neb_ispin"],
        "ISYM": values["neb_isym"],
        "ALGO": values["neb_algo"],
        "LREAL": values["neb_lreal"],
    }


# ----------------------------------------------------------------
# Phonon
# ----------------------------------------------------------------

#: ``(method, symmetry)`` to IBRION. DFPT gets the force constants from
#: perturbation theory in one run; finite differences displaces every
#: symmetry-inequivalent atom and reads the forces back. Keeping symmetry
#: is much cheaper and is only safe when the cell really has it.
PHONON_IBRION: dict[tuple[str, str], str] = {
    ("DFPT", "With symmetry"): "8",
    ("DFPT", "Without symmetry"): "7",
    ("Finite Difference", "With symmetry"): "6",
    ("Finite Difference", "Without symmetry"): "5",
}

PHONON = Job(
    name="Phonon Calculation",
    tagline="Force constants by DFPT or finite differences.",
    system="VASP Phonon",
    sections=(
        Section(
            "Phonon Calculation",
            (_system("phonon_system", "VASP Phonon"),),
        ),
        Section(
            "Phonon Method",
            (
                Field(
                    "phonon_method",
                    "Method",
                    Kind.CHOICE,
                    "DFPT",
                    choices=(
                        Choice("DFPT", "DFPT - linear response"),
                        Choice(
                            "Finite Difference",
                            "Finite Difference - displace and read forces",
                        ),
                    ),
                    note="How the force constants are obtained.",
                ),
                Field(
                    "phonon_symmetry",
                    "Symmetry",
                    Kind.CHOICE,
                    "With symmetry",
                    choices=(
                        Choice("With symmetry", "With symmetry"),
                        Choice("Without symmetry", "Without symmetry"),
                    ),
                    note=(
                        "Symmetry cuts the number of displacements, at the "
                        "cost of trusting the cell to have it."
                    ),
                ),
            ),
        ),
        Section(
            "Electronic Parameters",
            (
                _encut("phonon_encut"),
                _prec("phonon_prec", "Accurate"),
                _ediff(
                    "phonon_ediff",
                    "1E-8",
                ),
                _nelm("phonon_nelm"),
                _nelmin("phonon_nelmin", "5"),
                _algo("phonon_algo"),
                _ismear("phonon_ismear", "0"),
                _sigma("phonon_sigma"),
                _ispin("phonon_ispin"),
                _lreal(
                    "phonon_lreal",
                    LREAL_AUTO_FIRST,
                    FALSE,
                    note=(
                        "Reciprocal space. A phonon is a second derivative "
                        "and the real-space error shows up in it."
                    ),
                ),
            ),
        ),
        Section(
            "Finite Difference Parameters",
            (
                _potim_displacement("phonon_potim"),
                Field(
                    "phonon_nfree",
                    "NFREE",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("2", "2 - Central differences"),
                        Choice("4", "4 - Four-point differences"),
                    ),
                    note="Displacements per degree of freedom.",
                ),
                Field(
                    "phonon_nsw_fd",
                    "NSW",
                    Kind.INT,
                    "1",
                    low=1,
                    note="Must be positive for finite-difference phonons.",
                ),
                Field(
                    "phonon_checkpoint_fd",
                    "CHECKPOINT_FD",
                    Kind.CHOICE,
                    "RESET",
                    choices=(
                        Choice("RESET", "RESET - start clean, write no tag"),
                        Choice("NONE", "NONE - no checkpointing"),
                        Choice("PREPARE", "PREPARE - write the displacements"),
                        Choice("SINGLE", "SINGLE - run one displacement"),
                        Choice("CONTINUE", "CONTINUE - resume where it stopped"),
                    ),
                    note=(
                        "For splitting a large displacement set across "
                        "several jobs. RESET is the default and is not "
                        "written to the file."
                    ),
                ),
            ),
            when=(("phonon_method", ("Finite Difference",)),),
        ),
        Section(
            "DFPT Parameters",
            (
                Field(
                    "phonon_nsw_dfpt",
                    "NSW",
                    Kind.INT,
                    "1",
                    low=0,
                    note="The ions are not relaxed in a DFPT run.",
                ),
                _logical(
                    "phonon_lepsilon",
                    "LEPSILON",
                    FALSE,
                    note=(
                        "Dielectric tensor and Born effective charges. "
                        "Needed for the LO-TO splitting in a polar solid."
                    ),
                ),
            ),
            when=(("phonon_method", ("DFPT",)),),
        ),
        Section(
            "Ionic / Cell Response",
            (
                Field(
                    "phonon_isif_fd",
                    "ISIF",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("2", "2 - Force constants only"),
                        Choice(
                            "3",
                            "3 - Also the elastic and internal-strain tensors",
                        ),
                    ),
                    note="How much of the response is computed.",
                ),
            ),
            when=(("phonon_method", ("Finite Difference",)),),
        ),
        Section(
            "Ionic / Cell Response",
            (
                Field(
                    "phonon_isif_dfpt",
                    "ISIF",
                    Kind.FIXED,
                    "2",
                    note="Fixed cell.",
                ),
            ),
            when=(("phonon_method", ("DFPT",)),),
        ),
        Section(
            "Phonon Output",
            (
                _logical(
                    "phonon_lwave",
                    "LWAVE",
                    FALSE,
                    note=(
                        "A displacement set writes one of these per "
                        "displacement. Off unless a restart is planned."
                    ),
                ),
                _logical(
                    "phonon_lcharg",
                    "LCHARG",
                    FALSE,
                    note="Likewise one CHGCAR per displacement.",
                ),
            ),
        ),
        Section(
            "Phonon Dispersion",
            (
                _logical(
                    "phonon_lphon_dispersion",
                    "LPHON_DISPERSION",
                    FALSE,
                    note="Dispersion along a q-path given in QPOINTS.",
                ),
                Field(
                    "phonon_phon_nwrite",
                    "PHON_NWRITE",
                    Kind.INT,
                    "1",
                    low=0,
                    note="How much of the dispersion is written out.",
                ),
            ),
        ),
    ),
)


def phonon_ibrion(method: str, symmetry: str) -> str:
    """
    The mover for a phonon run, from the method and whether symmetry is kept.

    Falls back to finite differences without symmetry -- the slowest and
    most forgiving of the four -- if either name is unrecognised, since
    that is the combination least likely to give a wrong answer quietly.
    """

    return PHONON_IBRION.get((method, symmetry), "5")


def _build_phonon(setup: Setup) -> dict[str, str]:

    values = setup.values

    method = values["phonon_method"]
    symmetry = values["phonon_symmetry"]

    parameters = {
        "SYSTEM": values["phonon_system"],
        "ENCUT": values["phonon_encut"],
        "PREC": values["phonon_prec"],
        "EDIFF": values["phonon_ediff"],
        "NELM": values["phonon_nelm"],
        "NELMIN": values["phonon_nelmin"],
        "ALGO": values["phonon_algo"],
        "ISMEAR": values["phonon_ismear"],
        "SIGMA": values["phonon_sigma"],
        "ISPIN": values["phonon_ispin"],
        "LREAL": values["phonon_lreal"],
        "IBRION": phonon_ibrion(method, symmetry),
        "ISIF": (
            values["phonon_isif_fd"]
            if method == "Finite Difference"
            else values["phonon_isif_dfpt"]
        ),
        # ISYM is not offered as a control: it follows from the symmetry
        # choice above, and letting the two disagree is how a phonon run
        # ends up displacing atoms it has already declared equivalent.
        "ISYM": "2" if symmetry == "With symmetry" else "0",
        "LWAVE": values["phonon_lwave"],
        "LCHARG": values["phonon_lcharg"],
    }

    if method == "Finite Difference":

        parameters["POTIM"] = values["phonon_potim"]
        parameters["NFREE"] = values["phonon_nfree"]
        parameters["NSW"] = values["phonon_nsw_fd"]

        checkpoint = values["phonon_checkpoint_fd"]

        if checkpoint != "RESET":
            parameters["CHECKPOINT_FD"] = checkpoint

    else:

        parameters["NSW"] = values["phonon_nsw_dfpt"]

        if values["phonon_lepsilon"] == TRUE:
            parameters["LEPSILON"] = TRUE

    if values["phonon_lphon_dispersion"] == TRUE:
        parameters["LPHON_DISPERSION"] = TRUE
        parameters["PHON_NWRITE"] = values["phonon_phon_nwrite"]

    return parameters


# ----------------------------------------------------------------
# Elastic constants
# ----------------------------------------------------------------

ELASTIC = Job(
    name="Elastic Constant",
    tagline="The full elastic tensor by finite differences.",
    system="VASP Elastic Constants",
    sections=(
        Section(
            "Elastic Constant Calculation",
            (
                _system("elastic_system", "VASP Elastic Constants"),
                Field(
                    "elastic_method",
                    "Method",
                    Kind.CHOICE,
                    "Finite Difference",
                    choices=(
                        Choice("Finite Difference", "Finite Difference"),
                    ),
                    note=(
                        "VASP gets the elastic tensor by straining the cell "
                        "and differentiating the stress."
                    ),
                ),
            ),
        ),
        Section(
            "Elastic Response Parameters",
            (
                Field(
                    "elastic_ibrion",
                    "IBRION",
                    Kind.FIXED,
                    "6",
                    note="Finite differences with symmetry.",
                ),
                Field(
                    "elastic_isif",
                    "ISIF",
                    Kind.FIXED,
                    "3",
                    note="The cell has to be strained for a stress response.",
                ),
                Field(
                    "elastic_nfree",
                    "NFREE",
                    Kind.FIXED,
                    "4",
                    note="Displacements per degree of freedom.",
                ),
                Field(
                    "elastic_isym",
                    "ISYM",
                    Kind.FIXED,
                    "2",
                    note=(
                        "Symmetry is kept: it is what reduces the strain "
                        "patterns to the independent ones."
                    ),
                ),
                _potim_displacement("elastic_potim"),
                Field(
                    "elastic_nsw",
                    "NSW",
                    Kind.INT,
                    "1",
                    low=0,
                    note="Ionic steps used for the finite differences.",
                ),
            ),
        ),
        Section(
            "Electronic Parameters",
            (
                _encut("elastic_encut"),
                _prec(
                    "elastic_prec",
                    "Accurate",
                ),
                _ediff("elastic_ediff", "1E-8"),
                _nelm("elastic_nelm"),
                _nelmin("elastic_nelmin", "5"),
                _algo("elastic_algo"),
                _ismear("elastic_ismear", "0"),
                _sigma("elastic_sigma"),
                _ispin("elastic_ispin"),
                _lreal(
                    "elastic_lreal",
                    LREAL_FALSE_FIRST,
                    FALSE,
                    note=(
                        "Reciprocal space. The stress tensor is what is "
                        "being differentiated and it is sensitive to this."
                    ),
                ),
            ),
        ),
        Section(
            "Output",
            (
                _logical(
                    "elastic_lwave",
                    "LWAVE",
                    FALSE,
                    note="One per strain pattern. Off unless it is needed.",
                ),
                _logical(
                    "elastic_lcharg",
                    "LCHARG",
                    FALSE,
                    note="Likewise one per strain pattern.",
                ),
            ),
        ),
    ),
)


def _build_elastic(setup: Setup) -> dict[str, str]:

    values = setup.values

    return {
        "SYSTEM": values["elastic_system"],
        "ENCUT": values["elastic_encut"],
        "PREC": values["elastic_prec"],
        "IBRION": "6",
        "ISIF": "3",
        "NFREE": "3",
        "POTIM": values["elastic_potim"],
        "ISYM": "2",
        "NSW": values["elastic_nsw"],
        "EDIFF": values["elastic_ediff"],
        "NELM": values["elastic_nelm"],
        "NELMIN": values["elastic_nelmin"],
        "ALGO": values["elastic_algo"],
        "ISMEAR": values["elastic_ismear"],
        "SIGMA": values["elastic_sigma"],
        "ISPIN": values["elastic_ispin"],
        "LREAL": values["elastic_lreal"],
        "LWAVE": values["elastic_lwave"],
        "LCHARG": values["elastic_lcharg"],
    }


# ----------------------------------------------------------------
# Frequency
# ----------------------------------------------------------------

FREQUENCY = Job(
    name="Frequency Calculation",
    tagline="Vibrational modes at the Gamma point.",
    system="VASP Frequency Calculation",
    sections=(
        Section(
            "Frequency Calculation",
            (_system("frequency_system", "VASP Frequency Calculation"),),
        ),
        Section(
            "Vibrational Parameters",
            (
                Field(
                    "frequency_ibrion",
                    "IBRION",
                    Kind.FIXED,
                    "5",
                    note="Hessian from finite differences.",
                ),
                Field(
                    "frequency_nfree",
                    "NFREE",
                    Kind.FIXED,
                    "2",
                    note="Central differences.",
                ),
                Field(
                    "frequency_isif",
                    "ISIF",
                    Kind.FIXED,
                    "2",
                    note="Fixed cell.",
                ),
                _potim_displacement("frequency_potim"),
                Field(
                    "frequency_nsw",
                    "NSW",
                    Kind.INT,
                    "1",
                    low=0,
                    note="Ionic steps used for the finite differences.",
                ),
                Field(
                    "frequency_isym",
                    "ISYM",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("2", "2 - Symmetry on"),
                        Choice("0", "0 - Symmetry off for the charge density"),
                        Choice("-1", "-1 - Symmetry off entirely"),
                    ),
                    note=(
                        "Symmetry may hide modes that break it. Switch it "
                        "off if a soft mode is expected."
                    ),
                ),
            ),
        ),
        Section(
            "Electronic Parameters",
            (
                _encut("frequency_encut"),
                _prec("frequency_prec", "Accurate"),
                _ediff("frequency_ediff", "1E-8"),
                _nelm("frequency_nelm"),
                _nelmin("frequency_nelmin", "5"),
                _algo("frequency_algo"),
                _ismear("frequency_ismear", "0"),
                _sigma("frequency_sigma"),
                _ispin("frequency_ispin"),
                _lreal(
                    "frequency_lreal",
                    LREAL_FALSE_FIRST,
                    FALSE,
                    note=(
                        "Reciprocal space. The displacements here are "
                        "small and the forces have to be clean."
                    ),
                ),
            ),
        ),
    ),
)


def _build_frequency(setup: Setup) -> dict[str, str]:

    values = setup.values

    return {
        "SYSTEM": values["frequency_system"],
        "ENCUT": values["frequency_encut"],
        "PREC": values["frequency_prec"],
        "IBRION": "5",
        "NFREE": "2",
        "ISIF": "2",
        "POTIM": values["frequency_potim"],
        "NSW": values["frequency_nsw"],
        "ISYM": values["frequency_isym"],
        "EDIFF": values["frequency_ediff"],
        "NELM": values["frequency_nelm"],
        "NELMIN": values["frequency_nelmin"],
        "ALGO": values["frequency_algo"],
        "ISMEAR": values["frequency_ismear"],
        "SIGMA": values["frequency_sigma"],
        "ISPIN": values["frequency_ispin"],
        "LREAL": values["frequency_lreal"],
    }


# ----------------------------------------------------------------
# Optical
# ----------------------------------------------------------------

OPTICAL = Job(
    name="Optical Properties",
    tagline="Frequency-dependent dielectric function.",
    system="VASP Optical Properties",
    sections=(
        Section(
            "Optical Properties",
            (_system("optical_system", "VASP Optical Properties"),),
        ),
        Section(
            "Optical Calculation",
            (
                _logical(
                    "optical_loptics",
                    "LOPTICS",
                    TRUE,
                    note=(
                        "Compute the frequency-dependent dielectric matrix "
                        "in the independent-particle picture."
                    ),
                ),
                Field(
                    "optical_nedos",
                    "NEDOS",
                    Kind.INT,
                    "2000",
                    low=100,
                    step=100,
                    note="Points on the energy grid of the spectrum.",
                ),
                Field(
                    "optical_cshift",
                    "CSHIFT",
                    Kind.FLOAT,
                    "0.10",
                    low=0.0001,
                    step=0.01,
                    decimals=3,
                    note=(
                        "Complex shift, in eV. It broadens the spectrum; "
                        "too large a value smears real structure away."
                    ),
                ),
            ),
        ),
        Section(
            "Optical Tensor",
            (
                _lreal(
                    "optical_lreal",
                    LREAL_FALSE_FIRST,
                    FALSE,
                    note=(
                        "Reciprocal space. The transition matrix elements "
                        "want the accurate projectors."
                    ),
                ),
                Field(
                    "optical_lorbit",
                    "LORBIT",
                    Kind.CHOICE,
                    "11",
                    choices=LORBIT_CHOICES,
                    note="How much projection detail is written.",
                ),
                Field(
                    "optical_isym",
                    "ISYM",
                    Kind.CHOICE,
                    "2",
                    choices=(
                        Choice("2", "2 - Symmetry on"),
                        Choice("0", "0 - Symmetry off for the charge density"),
                        Choice("-1", "-1 - Symmetry off entirely"),
                    ),
                    note="Symmetry used to reduce the k-point set.",
                ),
            ),
        ),
        Section(
            "Electronic Parameters",
            (
                _encut("optical_encut"),
                _prec("optical_prec", "Accurate"),
                _ediff("optical_ediff", "1E-8"),
                _nelm("optical_nelm"),
                _nelmin("optical_nelmin", "5"),
                _algo("optical_algo"),
                _ismear("optical_ismear", "0"),
                _sigma("optical_sigma"),
                _ispin("optical_ispin"),
            ),
        ),
        Section(
            "Number of Bands",
            (
                Field(
                    "optical_nbands",
                    "NBANDS",
                    Kind.INT,
                    "100",
                    low=1,
                    note=(
                        "Bands in the transition sum. A spectrum needs "
                        "empty states well above the gap, so this is the "
                        "one number worth raising by hand."
                    ),
                ),
            ),
        ),
        Section(
            "Ionic Settings",
            (
                Field(
                    "optical_ibrion",
                    "IBRION",
                    Kind.FIXED,
                    "-1",
                    note="Static: no ionic relaxation.",
                ),
                Field(
                    "optical_nsw",
                    "NSW",
                    Kind.FIXED,
                    "0",
                    note="No ionic steps.",
                ),
                Field(
                    "optical_isif",
                    "ISIF",
                    Kind.FIXED,
                    "2",
                    note="Fixed cell.",
                ),
            ),
        ),
    ),
)


def _build_optical(setup: Setup) -> dict[str, str]:

    values = setup.values

    return {
        "SYSTEM": values["optical_system"],
        "ENCUT": values["optical_encut"],
        "PREC": values["optical_prec"],
        "LOPTICS": values["optical_loptics"],
        "NEDOS": values["optical_nedos"],
        "CSHIFT": values["optical_cshift"],
        "LORBIT": values["optical_lorbit"],
        "NBANDS": values["optical_nbands"],
        "EDIFF": values["optical_ediff"],
        "NELM": values["optical_nelm"],
        "NELMIN": values["optical_nelmin"],
        "ALGO": values["optical_algo"],
        "ISMEAR": values["optical_ismear"],
        "SIGMA": values["optical_sigma"],
        "ISPIN": values["optical_ispin"],
        "ISYM": values["optical_isym"],
        "LREAL": values["optical_lreal"],
        "IBRION": "-1",
        "NSW": "0",
        "ISIF": "2",
    }


# ----------------------------------------------------------------
# The eight jobs
# ----------------------------------------------------------------

JOBS: tuple[Job, ...] = (
    OPTIMIZATION,
    SINGLE_POINT,
    AIMD,
    NEB,
    PHONON,
    ELASTIC,
    FREQUENCY,
    OPTICAL,
)

JOBS_BY_NAME: dict[str, Job] = {job.name: job for job in JOBS}

BUILDERS = {
    "Optimization": _build_optimization,
    "Single-Point": _build_single_point,
    "AIMD": _build_aimd,
    "NEB": _build_neb,
    "Phonon Calculation": _build_phonon,
    "Elastic Constant": _build_elastic,
    "Frequency Calculation": _build_frequency,
    "Optical Properties": _build_optical,
}


def job_by_name(name: str) -> Job | None:
    """The job of that name, or ``None``."""

    return JOBS_BY_NAME.get(name)


# ----------------------------------------------------------------
# Optional physics
# ----------------------------------------------------------------

#: Effective U for the transition metals, in eV. These are the values
#: fitted against formation energies for the Dudarev scheme, which is
#: why LDAUTYPE is 2 and J is zero throughout: only U - J enters, and
#: putting the whole of it in U is the convention.
TRANSITION_METAL_U: dict[str, float] = {
    # 3d
    "Sc": 2.0,
    "Ti": 3.0,
    "V": 3.1,
    "Cr": 3.5,
    "Mn": 3.9,
    "Fe": 5.3,
    "Co": 3.3,
    "Ni": 6.2,
    "Cu": 4.0,
    "Zn": 6.0,
    # 4d
    "Y": 2.0,
    "Zr": 3.0,
    "Nb": 1.5,
    "Mo": 3.0,
    "Tc": 2.0,
    "Ru": 2.0,
    "Rh": 2.0,
    "Pd": 1.5,
    "Ag": 1.0,
    "Cd": 2.0,
    # 5d
    "Hf": 2.5,
    "Ta": 2.0,
    "W": 2.0,
    "Re": 2.0,
    "Os": 2.0,
    "Ir": 2.0,
    "Pt": 1.5,
    "Au": 1.5,
    "Hg": 2.0,
}

#: LDAUL for a species carrying the correction: 2 is the d shell.
D_SHELL = "2"

#: LDAUL for a species that carries none.
NO_SHELL = "-1"

#: Optional modules, in the order their blocks are written.
MODULES: tuple[str, ...] = (
    "DFT+U",
    "SOC",
    "GGA",
    "Meta-GGA",
    "IVDW",
    "HSE / Hybrid",
    "VASPsol",
)

#: What each does, shown on its switch.
MODULE_NOTES: dict[str, str] = {
    "DFT+U": "Hubbard correction on the localised d states.",
    "SOC": "Spin-orbit coupling, non-collinear.",
    "GGA": "Choose the exchange-correlation functional.",
    "Meta-GGA": "SCAN, which sees the kinetic energy density too.",
    "IVDW": "Dispersion correction.",
    "HSE / Hybrid": "Screened exact exchange.",
    "VASPsol": "Implicit solvation.",
}

MODULE_SECTIONS: dict[str, Section] = {
    "VASPsol": Section(
        "VASPsol Parameters",
        (
            _logical("vaspsol_lsol", "LSOL", TRUE, note="Switch solvation on."),
            Field(
                "vaspsol_eb_k",
                "EB_K",
                Kind.FLOAT,
                "78.4",
                low=1.0,
                step=0.1,
                decimals=1,
                note="Relative permittivity. 78.4 is water at 298 K.",
            ),
            Field(
                "vaspsol_tau",
                "TAU",
                Kind.FLOAT,
                "0.0",
                low=0.0,
                step=0.01,
                decimals=3,
                note=(
                    "Effective surface tension. Zero drops the cavitation "
                    "term, which is the usual choice."
                ),
            ),
        ),
    ),
    "SOC": Section(
        "Spin-Orbit Coupling",
        (
            Field(
                "soc_lsorbit",
                "LSORBIT",
                Kind.BOOL,
                TRUE,
                note="Include spin-orbit coupling.",
            ),
            Field(
                "soc_lnoncollinear",
                "LNONCOLLINEAR",
                Kind.BOOL,
                FALSE,
                note=(
                    "Fully non-collinear spins. LSORBIT already implies "
                    "this, so it is only set on its own."
                ),
            ),
            Field(
                "soc_saxis",
                "SAXIS",
                Kind.TEXT,
                "0 0 1",
                note="Quantisation axis, in Cartesian coordinates.",
            ),
        ),
    ),
    "GGA": Section(
        "GGA Parameters",
        (
            Field(
                "gga_functional",
                "GGA",
                Kind.CHOICE,
                "PE",
                choices=(
                    Choice("PE", "PE - PBE"),
                    Choice("PS", "PS - PBEsol"),
                    Choice("RP", "RP - revPBE"),
                    Choice("AM", "AM - AM05"),
                    Choice("PZ", "PZ - Perdew-Zunger LDA"),
                ),
                note="Which exchange-correlation functional is used.",
            ),
            Field(
                "gga_lasph",
                "LASPH",
                Kind.BOOL,
                TRUE,
                note=(
                    "Aspherical gradient corrections inside the PAW "
                    "spheres. Matters for anything with d or f electrons."
                ),
            ),
        ),
    ),
    "Meta-GGA": Section(
        "Meta-GGA Parameters",
        (
            Field(
                "meta_gga_functional",
                "METAGGA",
                Kind.CHOICE,
                "SCAN",
                choices=(
                    Choice("SCAN", "SCAN"),
                    Choice("RTPSS", "RTPSS - revised TPSS"),
                    Choice("TPSS", "TPSS"),
                    Choice("MS0", "MS0"),
                    Choice("MS1", "MS1"),
                    Choice("MS2", "MS2"),
                ),
                note="Which meta-GGA is used.",
            ),
            Field(
                "meta_gga_lasph",
                "LASPH",
                Kind.BOOL,
                TRUE,
                note="Required for a meta-GGA to be evaluated correctly.",
            ),
        ),
    ),
    "IVDW": Section(
        "van der Waals Parameters",
        (
            Field(
                "vdw_ivdw",
                "IVDW",
                Kind.CHOICE,
                "2",
                choices=(
                    Choice("0", "0 - No correction"),
                    Choice("1", "1 - DFT-D2, Grimme"),
                    Choice("2", "2 - DFT-D2, Grimme"),
                    Choice("11", "11 - DFT-D3, zero damping"),
                    Choice("12", "12 - DFT-D3, Becke-Johnson damping"),
                    Choice("20", "20 - Tkatchenko-Scheffler"),
                    Choice("21", "21 - Tkatchenko-Scheffler, iterative Hirshfeld"),
                ),
                note="Which dispersion correction is added.",
            ),
        ),
    ),
    "HSE / Hybrid": Section(
        "HSE / Hybrid Functional",
        (
            Field(
                "hse_type",
                "Functional",
                Kind.CHOICE,
                "HSE06",
                choices=(
                    Choice("HSE06", "HSE06 - AEXX 0.25, HFSCREEN 0.2"),
                    Choice("PBE0", "PBE0 - AEXX 0.25, unscreened"),
                    Choice("Custom", "Custom"),
                ),
                note=(
                    "Picking one sets the two numbers below; they stay "
                    "editable."
                ),
            ),
            Field(
                "hse_aexx",
                "AEXX",
                Kind.FLOAT,
                "0.25",
                low=0.0,
                high=1.0,
                step=0.05,
                decimals=2,
                note="Fraction of exact exchange.",
            ),
            Field(
                "hse_hfscreen",
                "HFSCREEN",
                Kind.FLOAT,
                "0.2",
                low=0.0,
                step=0.05,
                decimals=2,
                note=(
                    "Screening length, in A^-1. Zero is unscreened, which "
                    "is what makes PBE0 PBE0 rather than HSE."
                ),
            ),
        ),
    ),
}

#: What each hybrid is: the exchange fraction and the screening. Chosen
#: from the dropdown, then editable, because the whole point of "Custom"
#: is to move them.
HYBRID_PRESETS: dict[str, tuple[str, str]] = {
    "HSE06": ("0.25", "0.2"),
    "PBE0": ("0.25", "0.0"),
}


def hubbard_lists(
    elements, chosen, values: dict[str, str]
) -> tuple[str, str, str]:
    """
    LDAUL, LDAUU and LDAUJ, one entry per species in POSCAR order.

    The order is the whole difficulty. VASP reads these three lists
    positionally against the POTCAR, so a list written for the species
    that happen to carry a U -- rather than for every species -- does not
    fail. It silently applies the correction to the wrong element.

    J is zero throughout: this is the Dudarev scheme, where only the
    difference U - J enters and the convention is to put all of it in U.
    """

    shells: list[str] = []
    us: list[str] = []
    js: list[str] = []

    picked = set(chosen)

    for element in elements:

        if element in picked:

            shells.append(D_SHELL)
            us.append(values.get(f"dftu_u_{element}", "0"))
            js.append("0")

        else:

            shells.append(NO_SHELL)
            us.append("0")
            js.append("0")

    return " ".join(shells), " ".join(us), " ".join(js)


def transition_metals(elements) -> tuple[str, ...]:
    """Which of the species present have a tabulated U."""

    return tuple(
        element for element in elements if element in TRANSITION_METAL_U
    )


# ----------------------------------------------------------------
# Global parameters
# ----------------------------------------------------------------

GLOBAL_SECTIONS: tuple[Section, ...] = (
    Section(
        "Output & Charge Density",
        (
            _logical(
                "global_lcharg",
                "LCHARG",
                FALSE,
                note="Write CHGCAR and CHG.",
            ),
            _logical(
                "global_lwave",
                "LWAVE",
                FALSE,
                note="Write WAVECAR. Large, and only needed for a restart.",
            ),
            Field(
                "global_icharg",
                "ICHARG",
                Kind.CHOICE,
                "0",
                choices=ICHARG_CHOICES,
                note="Where the starting charge density comes from.",
            ),
        ),
    ),
    Section(
        "Parallelization",
        (
            Field(
                "global_kpar",
                "KPAR",
                Kind.INT,
                "1",
                low=1,
                note="k-point groups run in parallel.",
            ),
            Field(
                "global_ncore",
                "NCORE",
                Kind.INT,
                "1",
                low=1,
                note="Cores working on one orbital.",
            ),
            Field(
                "global_npar",
                "NPAR",
                Kind.INT,
                "1",
                low=1,
                note=(
                    "Bands treated in parallel. NCORE and NPAR are two "
                    "views of one split; setting both is asking for "
                    "trouble."
                ),
            ),
            _logical(
                "global_lplane",
                "LPLANE",
                TRUE,
                note="Distribute the FFTs by plane.",
            ),
            _logical(
                "global_lscalu",
                "LSCALU",
                FALSE,
                note="scaLAPACK for the LU decomposition.",
            ),
            Field(
                "global_nsim",
                "NSIM",
                Kind.INT,
                "4",
                low=1,
                note="Bands optimised at once in RMM-DIIS.",
            ),
        ),
    ),
    Section(
        "Electronic Mixing",
        (
            Field(
                "global_amix",
                "AMIX",
                Kind.FLOAT,
                "0.400",
                low=0.0,
                step=0.05,
                decimals=3,
                note="Linear mixing of the charge density.",
            ),
            Field(
                "global_bmix",
                "BMIX",
                Kind.FLOAT,
                "1.000",
                low=0.0,
                step=0.05,
                decimals=3,
                note="Kerker cutoff wavevector.",
            ),
            Field(
                "global_amix_mag",
                "AMIX_MAG",
                Kind.FLOAT,
                "1.600",
                low=0.0,
                step=0.05,
                decimals=3,
                note="Linear mixing of the magnetisation density.",
            ),
            Field(
                "global_bmix_mag",
                "BMIX_MAG",
                Kind.FLOAT,
                "1.000",
                low=0.0,
                step=0.05,
                decimals=3,
                note="Kerker cutoff for the magnetisation density.",
            ),
        ),
    ),
)

#: The order the global block is written in, which is also the order the
#: sections above declare their fields.
GLOBAL_TAGS: tuple[tuple[str, str], ...] = tuple(
    (field.name, field.label)
    for section in GLOBAL_SECTIONS
    for field in section.fields
)


def global_defaults() -> dict[str, str]:
    """Every global setting at its starting value."""

    return {
        field.name: field.default
        for section in GLOBAL_SECTIONS
        for field in section.fields
    }


def module_defaults() -> dict[str, str]:
    """Every optional-module setting at its starting value."""

    return {
        field.name: field.default
        for section in MODULE_SECTIONS.values()
        for field in section.fields
    }


# ----------------------------------------------------------------
# What the dialog hands over
# ----------------------------------------------------------------


@dataclass
class Setup:
    """
    Everything a job needs to become a file.

    The values are flat and keyed by field name, which is what makes the
    builders readable: each one is the same list of assignments as the
    tag block it produces.
    """

    #: Which of the eight.
    job: str

    #: Field name to value, as it would be written.
    values: dict[str, str] = _field(default_factory=dict)

    #: For Single-Point: which boxes are ticked.
    subjobs: tuple[str, ...] = ()

    #: Which optional modules are switched on.
    modules: tuple[str, ...] = ()

    #: Species in POSCAR order. Empty when nothing is open, in which case
    #: the per-species tags are left out rather than written short.
    elements: tuple[str, ...] = ()

    #: How many ions of each, in the same order.
    counts: tuple[int, ...] = ()

    #: Which species carry a Hubbard U.
    hubbard: tuple[str, ...] = ()

    def per_species(self, prefix: str) -> str:
        """
        A per-species tag, one number per element in POSCAR order.

        Empty when no structure is open. LANGEVIN_GAMMA is read
        positionally against the POTCAR, so a list that is short by one
        species does not fail -- it gives the last element somebody
        else's friction.
        """

        if not self.elements:
            return ""

        parts = []

        for element in self.elements:

            raw = self.values.get(f"{prefix}_{element}", "10.0")

            try:
                parts.append(f"{float(raw):.2f}")

            except ValueError:
                parts.append(raw)

        return " ".join(parts)

    def band_breakdown(self) -> tuple[tuple[str, int, int, str, int], ...]:
        """
        Per element: symbol, ion count, bands each, block, and the product.

        The dialog shows this under the band count because the estimate
        is only as good as the block classification behind it, and an
        element that came out as ``unknown`` should be visible rather
        than quietly contributing nothing.
        """

        rows = []

        for element, count in zip(self.elements, self.counts):

            bands, block = bands_for(element)

            rows.append((element, count, bands, block, count * bands))

        return tuple(rows)

    def nbands(self, doubled: bool = False) -> int | None:
        """
        Bands to ask for, or ``None`` when there is no structure to count.

        A rough sum over the valence orbitals each element brings. VASP's
        own default is roughly the electron count plus a margin, which is
        thin for anything reading empty states -- a DOS, an optical
        spectrum, a bonding analysis.
        """

        if not self.elements or not self.counts:
            return None

        total = sum(row[-1] for row in self.band_breakdown())

        if total <= 0:
            return None

        return total * COHP_BAND_FACTOR if doubled else total


# ----------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------


def _block(title: str, lines) -> list[str]:
    """A blank line, a heading, and the tags under it."""

    body = [line for line in lines if line]

    if not body:
        return []

    return ["", f"{COMMENT} {title}", *body]


def _tags(parameters: dict[str, str]) -> list[str]:

    return [
        f"{tag} = {value}"
        for tag, value in parameters.items()
        if value != ""
    ]


def module_lines(setup: Setup) -> list[str]:
    """
    The optional blocks, in :data:`MODULES` order.

    Each is a heading and its tags, and none of it appears unless the
    module was switched on.
    """

    values = setup.values
    on = set(setup.modules)

    lines: list[str] = []

    # --- Hubbard U ------------------------------------------------

    if "DFT+U" in on and setup.hubbard and setup.elements:

        shells, us, js = hubbard_lists(setup.elements, setup.hubbard, values)

        lines += _block(
            "DFT+U",
            [
                f"LDAU = {TRUE}",
                "LDAUTYPE = 2",
                f"LDAUL = {shells}",
                f"LDAUU = {us}",
                f"LDAUJ = {js}",
                # The default of 2 is only enough for an s-p system. A d
                # system needs 4 and an f system 6, or the total energy
                # oscillates and never settles.
                "LMAXMIX = 4",
            ],
        )

    # --- Spin-orbit -----------------------------------------------

    if "SOC" in on:

        body = []

        if values.get("soc_lsorbit", TRUE) == TRUE:
            body.append(f"LSORBIT = {TRUE}")

        if values.get("soc_lnoncollinear", FALSE) == TRUE:
            body.append(f"LNONCOLLINEAR = {TRUE}")

        body.append(f"SAXIS = {values.get('soc_saxis', '0 0 1')}")

        lines += _block("SOC", body)

    # --- GGA ------------------------------------------------------

    if "GGA" in on:

        body = [f"GGA = {values.get('gga_functional', 'PE')}"]

        if values.get("gga_lasph", TRUE) == TRUE:
            body.append(f"LASPH = {TRUE}")

        lines += _block("GGA", body)

    # --- Meta-GGA -------------------------------------------------

    if "Meta-GGA" in on:

        body = [f"METAGGA = {values.get('meta_gga_functional', 'SCAN')}"]

        if values.get("meta_gga_lasph", TRUE) == TRUE:
            body.append(f"LASPH = {TRUE}")

        lines += _block("Meta-GGA", body)

    # --- Dispersion -----------------------------------------------

    if "IVDW" in on:
        lines += _block(
            "van der Waals", [f"IVDW = {values.get('vdw_ivdw', '2')}"]
        )

    # --- Hybrid ---------------------------------------------------

    if "HSE / Hybrid" in on:
        lines += _block(
            "HSE / Hybrid",
            [
                f"LHFCALC = {TRUE}",
                f"AEXX = {values.get('hse_aexx', '0.25')}",
                f"HFSCREEN = {values.get('hse_hfscreen', '0.2')}",
            ],
        )

    # --- Solvation ------------------------------------------------

    if "VASPsol" in on:
        lines += _block(
            "VASPsol",
            [
                f"LSOL = {values.get('vaspsol_lsol', TRUE)}",
                f"EB_K = {values.get('vaspsol_eb_k', '78.4')}",
                f"TAU = {values.get('vaspsol_tau', '0.0')}",
            ],
        )

    return lines


#: Files a sub-job cannot do without, and the tag that writes each. A
#: Bader partition needs a charge density to partition; LOBSTER needs the
#: wavefunctions to project. Both are off by default in the global block,
#: and the global block is written last.
REQUIRED_OUTPUT: dict[str, tuple[str, str]] = {
    "Bader Charge": ("LCHARG", TRUE),
    "Charge Density": ("LCHARG", TRUE),
    "COHP / ICOHP": ("LWAVE", TRUE),
}


def requirements(setup: Setup) -> dict[str, str]:
    """
    Tags the chosen sub-jobs cannot work without.

    These are the one exception to the global block having the last word.
    Someone who ticks "Bader Charge" and leaves LCHARG at its default has
    asked for two incompatible things, and of the two the sub-job is the
    one they chose deliberately -- so it wins, rather than the run
    finishing and leaving no CHGCAR to analyse.
    """

    forced: dict[str, str] = {}

    for subjob in setup.subjobs:

        entry = REQUIRED_OUTPUT.get(subjob)

        if entry is not None:
            forced[entry[0]] = entry[1]

    return forced


def global_lines(setup: Setup) -> list[str]:
    """
    The output, parallelisation and mixing block.

    Anything a sub-job requires is left out here rather than written and
    then overridden, so the file carries one line per tag and it is the
    line that survives.
    """

    values = setup.values
    forced = requirements(setup)

    body = [
        f"{label} = {values[name]}"
        for name, label in GLOBAL_TAGS
        if name in values and label not in forced
    ]

    return _block("Output / Charge Density", body)


def deduplicate(lines) -> list[str]:
    """
    One line per tag: the position of its first appearance, the value of
    its last.

    Both halves of that rule matter. Keeping the last value is what lets
    the global block override LWAVE that the job block already set, and
    what lets a COHP run raise the LORBIT a plain PDOS asked for. Keeping
    the first position is what stops the file being reordered every time
    a switch is flicked, so a diff between two INCARs shows the change
    and nothing else.
    """

    latest: dict[str, str] = {}

    for line in lines:

        stripped = line.strip()

        if not stripped or stripped.startswith(COMMENT) or "=" not in line:
            continue

        latest[line.split("=", 1)[0].strip().upper()] = line

    kept: list[str] = []
    written: set[str] = set()

    for line in lines:

        stripped = line.strip()

        if not stripped or stripped.startswith(COMMENT) or "=" not in line:
            kept.append(line)
            continue

        tag = line.split("=", 1)[0].strip().upper()

        if tag in written:
            continue

        kept.append(latest[tag])
        written.add(tag)

    return kept


def incar_text(setup: Setup) -> str:
    """
    The INCAR as it stands.

    Job first, then whatever optional physics was switched on, then the
    global block. Duplicates are resolved by :func:`deduplicate`, and
    what comes back is the file, ending in a newline.
    """

    job = job_by_name(setup.job)

    if job is None:
        return ""

    lines = _tags(BUILDERS[job.name](setup))

    if job.name == "Single-Point" and setup.subjobs:

        # The sub-jobs leave no trace in the tags -- a DOS and a Bader run
        # differ by a couple of switches -- so the file says which it was
        # meant to be. VASP ignores it; the next reader will not.
        lines += [
            "",
            f"{COMMENT} Single-Point Calculations",
            f"{COMMENT} " + ", ".join(setup.subjobs),
        ]

        forced = requirements(setup)

        if forced:

            # Say so in the file. These are the tags the selection took
            # out of the user's hands, and finding that out by reading
            # the generator is not reasonable.
            lines.append(
                f"{COMMENT} Required by the above: "
                + ", ".join(
                    f"{tag} = {value}" for tag, value in forced.items()
                )
            )

    lines += module_lines(setup)
    lines += global_lines(setup)

    text = "\n".join(deduplicate(lines)).strip()

    return text + "\n" if text else ""
