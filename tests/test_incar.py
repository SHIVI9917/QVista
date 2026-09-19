"""
INCAR generation.

The tests are about what ends up in the file. A wrong INCAR does not
usually fail -- VASP reads it, runs for a day and hands back an answer
that is merely incorrect -- so the things worth pinning down are the ones
where nothing complains: a tag that silently keeps the wrong value after
being set twice, per-species lists written against the wrong species
order, an ISYM that disagrees with the symmetry the user asked for.
"""

from __future__ import annotations

import pytest

from qvista.services import incar as ic


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def setup_for(name: str, **overrides) -> ic.Setup:
    """A job at its defaults, with anything named here changed."""

    job = ic.job_by_name(name)

    assert job is not None, name

    values = job.defaults() | ic.global_defaults() | ic.module_defaults()

    subjobs = overrides.pop("subjobs", ())
    modules = overrides.pop("modules", ())
    elements = overrides.pop("elements", ())
    counts = overrides.pop("counts", ())
    hubbard = overrides.pop("hubbard", ())

    values.update(overrides)

    return ic.Setup(
        job=name,
        values=values,
        subjobs=subjobs,
        modules=modules,
        elements=elements,
        counts=counts,
        hubbard=hubbard,
    )


def tags_of(text: str) -> dict[str, str]:
    """The ``TAG = value`` lines of an INCAR, as a mapping."""

    found: dict[str, str] = {}

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped or stripped.startswith(ic.COMMENT):
            continue

        tag, _, value = line.partition("=")

        found[tag.strip()] = value.strip()

    return found


def written(name: str, **overrides) -> dict[str, str]:

    return tags_of(ic.incar_text(setup_for(name, **overrides)))


# ----------------------------------------------------------------
# The eight jobs
# ----------------------------------------------------------------


def test_there_are_eight_jobs_and_every_one_has_a_builder():

    assert len(ic.JOBS) == 8

    for job in ic.JOBS:
        assert job.name in ic.BUILDERS


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_every_job_produces_a_file(job):

    text = ic.incar_text(setup_for(job, subjobs=("Static SCF",)))

    assert text.endswith("\n")
    assert tags_of(text)


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_every_job_names_itself(job):

    assert written(job, subjobs=("Static SCF",))["SYSTEM"]


def test_an_unknown_job_produces_nothing_rather_than_a_broken_file():

    assert ic.incar_text(ic.Setup(job="Band Unfolding")) == ""


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_no_field_name_is_used_twice_within_a_job(job):

    definition = ic.job_by_name(job)

    names = [field.name for field in definition.fields()]

    assert len(names) == len(set(names))


def test_no_field_name_collides_across_jobs_modules_or_globals():
    """
    Every setting of every job lives in one flat mapping, so that
    clicking away from a job and back does not lose what was typed. That
    only works while the names are unique.
    """

    seen: set[str] = set()

    for job in ic.JOBS:

        names = set(job.defaults())

        assert not (names & seen), job.name

        seen |= names

    for group in (ic.global_defaults(), ic.module_defaults()):

        assert not (set(group) & seen)

        seen |= set(group)


# ----------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------


def test_a_tag_set_twice_keeps_the_later_value():
    """
    The global block has the last word. Without that a job that writes
    LWAVE for its own reasons could not be overridden from the one place
    the user expects to control it.
    """

    text = ic.incar_text(
        setup_for("AIMD", aimd_lwave=ic.FALSE, global_lwave=ic.TRUE)
    )

    assert tags_of(text)["LWAVE"] == ic.TRUE

    # ...and it says it once, not twice.
    assert text.count("LWAVE =") == 1


def test_a_tag_set_twice_keeps_its_earlier_position():
    """
    Overriding a value must not reorder the file, or a diff between two
    INCARs shows every line as changed.
    """

    lines = ic.deduplicate(
        ["ENCUT = 520", "LWAVE = .FALSE.", "SIGMA = 0.05", "LWAVE = .TRUE."]
    )

    assert lines == ["ENCUT = 520", "LWAVE = .TRUE.", "SIGMA = 0.05"]


def test_comments_and_blank_lines_survive_deduplication():

    lines = ic.deduplicate(["ENCUT = 520", "", "# GGA", "GGA = PE"])

    assert lines == ["ENCUT = 520", "", "# GGA", "GGA = PE"]


def test_a_tag_is_matched_regardless_of_how_it_was_capitalised():

    lines = ic.deduplicate(["encut = 400", "ENCUT = 520"])

    assert lines == ["ENCUT = 520"]


def test_an_empty_value_is_left_out_rather_than_written_bare():
    """``LANGEVIN_GAMMA =`` with nothing after it is not a default."""

    assert "LANGEVIN_GAMMA" not in written(
        "AIMD", aimd_ensemble="NPT", elements=(), counts=()
    )


# ----------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------


def test_optimization_writes_the_mover_and_what_may_relax():

    tags = written("Optimization")

    assert tags["IBRION"] == "2"
    assert tags["ISIF"] == "2"
    assert tags["EDIFFG"] == "-0.02"


def test_a_relaxation_of_the_cell_is_available():

    assert written("Optimization", optimization_isif="3")["ISIF"] == "3"


# ----------------------------------------------------------------
# Single-Point
# ----------------------------------------------------------------


def test_a_single_point_never_moves_the_ions():
    """
    Whatever is being read off it, a single point is one geometry. If
    NSW came through from somewhere the run would relax instead, and the
    DOS would belong to a structure nobody asked about.
    """

    tags = written("Single-Point", subjobs=("Static SCF",))

    assert tags["IBRION"] == "-1"
    assert tags["NSW"] == "0"


def test_the_sub_jobs_are_recorded_in_the_file():
    """
    A DOS run and a Bader run differ by two switches, so the file says
    which it was meant to be.
    """

    text = ic.incar_text(setup_for("Single-Point", subjobs=("DOS", "ELF")))

    assert "# Single-Point Calculations" in text
    assert "# DOS, ELF" in text


def test_projections_are_only_asked_for_when_something_reads_them():

    assert "LORBIT" not in written("Single-Point", subjobs=("Static SCF",))
    assert "LORBIT" in written("Single-Point", subjobs=("PDOS",))


@pytest.mark.parametrize("subjob", ic.PROJECTED)
def test_every_projected_sub_job_asks_for_the_projections(subjob):

    tags = written("Single-Point", subjobs=(subjob,))

    assert "LORBIT" in tags
    assert "NEDOS" in tags


def test_a_bonding_analysis_overrides_a_plain_projection():
    """
    LOBSTER is stricter than a PDOS about what it needs. Asked for both,
    the file has to satisfy the stricter one.
    """

    tags = written(
        "Single-Point",
        subjobs=("PDOS", "COHP / ICOHP"),
        sp_lorbit="10",
        sp_cohp_lorbit="12",
    )

    assert tags["LORBIT"] == "12"
    assert tags["LWAVE"] == ic.TRUE


def test_a_bader_run_asks_for_the_all_electron_density():
    """
    A Bader partition of the pseudo-density alone gives charges that
    look plausible and are wrong, because the core is missing from it.
    """

    tags = written("Single-Point", subjobs=("Bader Charge",))

    assert tags["LAECHG"] == ic.TRUE
    assert tags["LCHARG"] == ic.TRUE


def test_a_required_output_survives_the_global_default():
    """
    The one place the global block does not get the last word.

    LCHARG and LWAVE are off by default and the global block is written
    last, so a Bader or COHP run assembled naively finishes and leaves
    nothing to analyse -- no error, no warning, just a missing file that
    is only noticed by the tool downstream.
    """

    bader = written("Single-Point", subjobs=("Bader Charge",))

    assert bader["LCHARG"] == ic.TRUE

    cohp = written("Single-Point", subjobs=("COHP / ICOHP",))

    assert cohp["LWAVE"] == ic.TRUE


def test_a_required_output_is_written_once():
    """Forced, not overridden: one line, not two disagreeing ones."""

    text = ic.incar_text(setup_for("Single-Point", subjobs=("Bader Charge",)))

    settings = [
        line
        for line in text.splitlines()
        if line.strip().startswith("LCHARG")
    ]

    assert settings == ["LCHARG = .TRUE."]


def test_a_required_output_is_declared_in_the_file():
    """
    The selection took the tag out of the user's hands. Finding that out
    by reading the generator is not reasonable.
    """

    text = ic.incar_text(setup_for("Single-Point", subjobs=("Bader Charge",)))

    assert "# Required by the above: LCHARG = .TRUE." in text


def test_nothing_is_forced_when_nothing_needs_it():

    assert requirements_for(("Static SCF", "DOS")) == {}

    text = ic.incar_text(setup_for("Single-Point", subjobs=("DOS",)))

    assert "Required by the above" not in text


def test_the_global_default_still_rules_the_tags_nobody_needs():

    tags = written(
        "Single-Point",
        subjobs=("Bader Charge",),
        global_lwave=ic.TRUE,
    )

    assert tags["LWAVE"] == ic.TRUE


def requirements_for(subjobs) -> dict[str, str]:

    return ic.requirements(ic.Setup(job="Single-Point", subjobs=subjobs))


def test_the_work_function_asks_for_the_potential():

    tags = written("Single-Point", subjobs=("Work Function",))

    assert tags["LVHAR"] == ic.TRUE
    assert tags["LVTOT"] == ic.TRUE


def test_elf_is_only_written_when_asked_for():

    assert "LELF" not in written("Single-Point", subjobs=("DOS",))
    assert written("Single-Point", subjobs=("ELF",))["LELF"] == ic.TRUE


def test_the_sections_on_show_follow_the_boxes_ticked():

    assert ic.single_point_sections(()) == (
        "Common Single-Point Parameters",
    )

    assert "Electronic Structure Parameters" in ic.single_point_sections(
        ("DOS",)
    )

    assert "Bader Charge Parameters" in ic.single_point_sections(
        ("Bader Charge",)
    )


# ----------------------------------------------------------------
# Band counting
# ----------------------------------------------------------------


def test_bands_are_counted_by_block():

    assert ic.bands_for("Na") == (1, "s-block")
    assert ic.bands_for("O") == (4, "p-block")
    assert ic.bands_for("Ni") == (9, "d-block")


def test_an_unclassified_element_is_reported_rather_than_guessed():
    """
    The f-block is not covered. Returning zero puts the gap in front of
    the user in the breakdown instead of hiding it in the total.
    """

    assert ic.bands_for("Ce") == (0, "unknown")


def test_the_band_count_is_the_sum_over_the_cell():

    setup = setup_for("Single-Point", elements=("Ni", "O"), counts=(4, 4))

    # 4 x 9 for the d-block nickel, 4 x 4 for the p-block oxygen.
    assert setup.nbands() == 52


def test_a_bonding_analysis_doubles_the_band_count():
    """
    COHP projects onto a local basis and needs empty states well above
    the Fermi level to do it.
    """

    setup = setup_for("Single-Point", elements=("Ni", "O"), counts=(4, 4))

    assert setup.nbands(doubled=True) == 104


def test_the_band_count_is_left_out_without_a_structure():
    """
    There is nothing to count. Writing a guess would cap the spectrum at
    a number that came from nowhere.
    """

    assert setup_for("Single-Point").nbands() is None
    assert "NBANDS" not in written("Single-Point", subjobs=("DOS",))


def test_the_band_count_reaches_the_file():

    tags = written(
        "Single-Point",
        subjobs=("DOS",),
        elements=("Ni", "O"),
        counts=(4, 4),
    )

    assert tags["NBANDS"] == "52"


def test_the_breakdown_accounts_for_every_species():

    setup = setup_for("Single-Point", elements=("Ni", "O"), counts=(4, 4))

    rows = setup.band_breakdown()

    assert [row[0] for row in rows] == ["Ni", "O"]
    assert sum(row[-1] for row in rows) == setup.nbands()


# ----------------------------------------------------------------
# AIMD
# ----------------------------------------------------------------


def test_molecular_dynamics_is_always_ibrion_zero():

    assert written("AIMD")["IBRION"] == "0"


def test_symmetry_is_off_for_dynamics():
    """
    The trajectory breaks the symmetry of the starting cell on the first
    step. Leaving ISYM on symmetrises forces that are no longer related.
    """

    assert written("AIMD")["ISYM"] in {"0", "-1"}


@pytest.mark.parametrize(
    "thermostat,mdalgo",
    [
        ("Nose-Hoover", "2"),
        ("Andersen", "1"),
        ("Langevin", "3"),
        ("Nose-Hoover Chain", "4"),
        ("CSVR", "5"),
    ],
)
def test_each_thermostat_selects_its_own_integrator(thermostat, mdalgo):

    tags = written(
        "AIMD", aimd_ensemble="NVT", aimd_nvt_thermostat=thermostat
    )

    assert tags["MDALGO"] == mdalgo


def test_a_thermostat_brings_only_its_own_settings():
    """
    SMASS means nothing to the Andersen integrator, and a leftover from
    a previously chosen thermostat is exactly the sort of tag that is
    read and quietly ignored.
    """

    tags = written(
        "AIMD", aimd_ensemble="NVT", aimd_nvt_thermostat="Andersen"
    )

    assert "ANDERSEN_PROB" in tags
    assert "SMASS" not in tags
    assert "CSVR_PERIOD" not in tags
    assert "NHC_NCHAINS" not in tags


def test_nvt_keeps_the_cell_fixed():

    assert int(written("AIMD", aimd_ensemble="NVT")["ISIF"]) < 3


def test_nve_switches_the_thermostat_off_rather_than_leaving_it_out():
    """
    VASP has no "no thermostat" setting. NVE is a thermostatted
    integrator with its coupling turned off, and the tag that turns it
    off has to be there.
    """

    tags = written(
        "AIMD",
        aimd_ensemble="NVE",
        aimd_nve_method="Andersen propagation",
    )

    assert tags["MDALGO"] == "1"
    assert tags["ANDERSEN_PROB"] == "0.0"

    tags = written(
        "AIMD",
        aimd_ensemble="NVE",
        aimd_nve_method="Nose-Hoover propagation",
    )

    assert tags["MDALGO"] == "2"
    assert tags["SMASS"] == "-3"


def test_npt_frees_the_cell_and_drives_it():

    tags = written(
        "AIMD",
        aimd_ensemble="NPT",
        elements=("Li", "O"),
        counts=(2, 2),
    )

    assert tags["ISIF"] == "3"
    assert tags["MDALGO"] == "3"
    assert "LANGEVIN_GAMMA_L" in tags
    assert "PMASS" in tags
    assert "PSTRESS" in tags


def test_the_friction_is_written_once_per_species_in_poscar_order():
    """
    LANGEVIN_GAMMA is read positionally against the POTCAR. A list that
    is short by one species is not rejected -- it gives the last element
    somebody else's friction.
    """

    setup = setup_for(
        "AIMD",
        aimd_ensemble="NPT",
        elements=("Li", "Ni", "O"),
        counts=(2, 2, 4),
        aimd_npt_gamma_Li="1.0",
        aimd_npt_gamma_Ni="5.0",
        aimd_npt_gamma_O="10.0",
    )

    assert tags_of(ic.incar_text(setup))["LANGEVIN_GAMMA"] == (
        "1.00 5.00 10.00"
    )


def test_the_friction_is_left_out_when_the_species_order_is_unknown():

    assert "LANGEVIN_GAMMA" not in written("AIMD", aimd_ensemble="NPT")


def test_a_langevin_thermostat_in_nvt_also_needs_one_per_species():

    setup = setup_for(
        "AIMD",
        aimd_ensemble="NVT",
        aimd_nvt_thermostat="Langevin",
        elements=("Li", "O"),
        counts=(2, 2),
        aimd_nvt_gamma_Li="3.0",
        aimd_nvt_gamma_O="7.0",
    )

    assert tags_of(ic.incar_text(setup))["LANGEVIN_GAMMA"] == "3.00 7.00"


# ----------------------------------------------------------------
# NEB
# ----------------------------------------------------------------


def test_a_band_is_optimised_with_damped_dynamics_and_a_fixed_cell():

    tags = written("NEB")

    assert tags["IBRION"] == "3"
    assert tags["POTIM"] == "0"
    assert tags["ISIF"] == "2"


def test_the_image_count_excludes_the_endpoints():
    """
    VASP's IMAGES counts the images it relaxes, not the folders on disk.
    Five images means seven folders, 00 to 06.
    """

    assert written("NEB", neb_images="5")["IMAGES"] == "5"


def test_a_band_converges_on_forces():

    assert written("NEB")["EDIFFG"].startswith("-")


# ----------------------------------------------------------------
# Phonon
# ----------------------------------------------------------------


@pytest.mark.parametrize(
    "method,symmetry,ibrion",
    [
        ("DFPT", "With symmetry", "8"),
        ("DFPT", "Without symmetry", "7"),
        ("Finite Difference", "With symmetry", "6"),
        ("Finite Difference", "Without symmetry", "5"),
    ],
)
def test_the_phonon_mover_follows_the_method_and_the_symmetry(
    method, symmetry, ibrion
):

    assert ic.phonon_ibrion(method, symmetry) == ibrion

    tags = written(
        "Phonon Calculation",
        phonon_method=method,
        phonon_symmetry=symmetry,
    )

    assert tags["IBRION"] == ibrion


def test_an_unrecognised_phonon_combination_falls_back_to_the_slow_one():
    """
    Finite differences without symmetry is the most expensive of the four
    and the least likely to be quietly wrong, which is the right way to
    fail.
    """

    assert ic.phonon_ibrion("", "") == "5"


def test_isym_agrees_with_the_symmetry_that_was_asked_for():
    """
    The two cannot be set independently. A run told to keep symmetry
    while ISYM says otherwise displaces atoms it has already declared
    equivalent, and the force constants come back short.
    """

    with_symmetry = written(
        "Phonon Calculation", phonon_symmetry="With symmetry"
    )
    without = written(
        "Phonon Calculation", phonon_symmetry="Without symmetry"
    )

    assert with_symmetry["ISYM"] == "2"
    assert without["ISYM"] == "0"


def test_a_phonon_run_produces_a_complete_file():
    """
    Guards a mismatch that is easy to introduce and silent: the settings
    being stored under one name and read back under another, which leaves
    the file empty of everything but the global block.
    """

    tags = written("Phonon Calculation")

    for tag in ("SYSTEM", "ENCUT", "PREC", "EDIFF", "IBRION", "ISIF"):
        assert tag in tags


def test_the_finite_difference_settings_only_appear_for_that_method():

    fd = written(
        "Phonon Calculation", phonon_method="Finite Difference"
    )
    dfpt = written("Phonon Calculation", phonon_method="DFPT")

    assert "NFREE" in fd
    assert "POTIM" in fd

    assert "NFREE" not in dfpt
    assert "POTIM" not in dfpt


def test_the_default_checkpoint_is_not_written():
    """
    RESET is what VASP does anyway. Writing it adds a tag whose presence
    suggests the run was split when it was not.
    """

    assert "CHECKPOINT_FD" not in written(
        "Phonon Calculation", phonon_method="Finite Difference"
    )

    assert (
        written(
            "Phonon Calculation",
            phonon_method="Finite Difference",
            phonon_checkpoint_fd="CONTINUE",
        )["CHECKPOINT_FD"]
        == "CONTINUE"
    )


def test_born_charges_are_only_asked_for_when_wanted():

    assert "LEPSILON" not in written("Phonon Calculation")

    assert (
        written("Phonon Calculation", phonon_lepsilon=ic.TRUE)["LEPSILON"]
        == ic.TRUE
    )


def test_a_phonon_uses_reciprocal_space_projectors():
    """
    A phonon is a second derivative, and the real-space projection error
    lands squarely in it.
    """

    assert written("Phonon Calculation")["LREAL"] == ic.FALSE


def test_the_dispersion_tags_travel_together():

    assert "PHON_NWRITE" not in written("Phonon Calculation")

    tags = written(
        "Phonon Calculation", phonon_lphon_dispersion=ic.TRUE
    )

    assert tags["LPHON_DISPERSION"] == ic.TRUE
    assert "PHON_NWRITE" in tags


# ----------------------------------------------------------------
# Elastic and frequency
# ----------------------------------------------------------------


def test_the_elastic_tensor_needs_a_strainable_cell():

    tags = written("Elastic Constant")

    assert tags["IBRION"] == "6"
    assert tags["ISIF"] == "3"
    assert tags["NFREE"] == "3"


def test_a_frequency_run_keeps_the_cell_fixed():

    tags = written("Frequency Calculation")

    assert tags["IBRION"] == "5"
    assert tags["ISIF"] == "2"
    assert tags["NFREE"] == "2"


@pytest.mark.parametrize(
    "job", ["Elastic Constant", "Frequency Calculation", "Phonon Calculation"]
)
def test_anything_differentiating_forces_converges_tightly(job):
    """
    A finite difference of a quantity converged to 1E-5 is noise. These
    three all differentiate, and all start at 1E-8.
    """

    assert written(job)["EDIFF"] == "1E-8"


@pytest.mark.parametrize(
    "job", ["Elastic Constant", "Frequency Calculation", "Optical Properties"]
)
def test_the_accurate_grids_are_the_default_where_they_matter(job):

    assert written(job)["PREC"] == "Accurate"


# ----------------------------------------------------------------
# Optical
# ----------------------------------------------------------------


def test_an_optical_run_is_static():

    tags = written("Optical Properties")

    assert tags["IBRION"] == "-1"
    assert tags["NSW"] == "0"


def test_the_dielectric_function_is_asked_for_by_default():

    tags = written("Optical Properties")

    assert tags["LOPTICS"] == ic.TRUE
    assert "CSHIFT" in tags
    assert "NEDOS" in tags


# ----------------------------------------------------------------
# Optional physics
# ----------------------------------------------------------------


def test_nothing_optional_appears_until_it_is_switched_on():

    text = ic.incar_text(setup_for("Optimization"))

    for absent in ("LDAU", "LSORBIT", "METAGGA", "IVDW", "LHFCALC", "LSOL"):
        assert absent not in text


def test_the_hubbard_lists_cover_every_species_in_poscar_order():
    """
    The heart of it. LDAUL, LDAUU and LDAUJ are read positionally against
    the POTCAR. A list written only for the species that carry a U is
    not rejected -- it applies the correction to whatever happens to sit
    in that slot, and the run completes.
    """

    setup = setup_for(
        "Optimization",
        modules=("DFT+U",),
        elements=("Li", "Ni", "O"),
        counts=(4, 4, 8),
        hubbard=("Ni",),
        dftu_u_Ni="6.2",
    )

    tags = tags_of(ic.incar_text(setup))

    assert tags["LDAUL"] == "-1 2 -1"
    assert tags["LDAUU"] == "0 6.2 0"
    assert tags["LDAUJ"] == "0 0 0"


def test_the_hubbard_lists_are_as_long_as_the_species_count():

    for count in range(1, 5):

        elements = tuple("ABCD"[:count])

        shells, us, js = ic.hubbard_lists(elements, (), {})

        assert len(shells.split()) == count
        assert len(us.split()) == count
        assert len(js.split()) == count


def test_a_hubbard_run_raises_lmaxmix():
    """
    The default of 2 is enough for an s-p system only. Left alone on a d
    system the total energy oscillates and the run never converges --
    slowly, and without ever saying why.
    """

    setup = setup_for(
        "Optimization",
        modules=("DFT+U",),
        elements=("Ni", "O"),
        counts=(4, 4),
        hubbard=("Ni",),
    )

    assert tags_of(ic.incar_text(setup))["LMAXMIX"] == "4"


def test_the_hubbard_block_is_left_out_without_a_species_order():
    """
    Switching +U on with nothing open cannot produce correct lists, so
    it produces none. A half-written LDAU block is worse than no block.
    """

    text = ic.incar_text(setup_for("Optimization", modules=("DFT+U",)))

    assert "LDAU" not in text


def test_only_the_transition_metals_present_are_offered():

    assert ic.transition_metals(("Li", "Ni", "O", "Fe")) == ("Ni", "Fe")
    assert ic.transition_metals(("Na", "Cl")) == ()


def test_the_hubbard_table_has_a_value_for_each_d_block_row():

    for element in ("Ti", "Fe", "Ni", "Mo", "Ru", "W", "Pt"):
        assert ic.TRANSITION_METAL_U[element] > 0


def test_spin_orbit_writes_the_axis_it_is_quantised_along():

    tags = tags_of(
        ic.incar_text(setup_for("Optimization", modules=("SOC",)))
    )

    assert tags["LSORBIT"] == ic.TRUE
    assert tags["SAXIS"] == "0 0 1"


def test_spin_orbit_can_be_declined_while_the_axis_is_kept():
    """
    The switch on LSORBIT is a switch, not decoration. Writing
    ``.TRUE.`` regardless would make it impossible to turn off.
    """

    tags = tags_of(
        ic.incar_text(
            setup_for(
                "Optimization", modules=("SOC",), soc_lsorbit=ic.FALSE
            )
        )
    )

    assert "LSORBIT" not in tags


def test_non_collinear_is_only_written_when_asked_for():

    plain = ic.incar_text(setup_for("Optimization", modules=("SOC",)))

    assert "LNONCOLLINEAR" not in plain

    asked = ic.incar_text(
        setup_for(
            "Optimization", modules=("SOC",), soc_lnoncollinear=ic.TRUE
        )
    )

    assert "LNONCOLLINEAR" in asked


def test_the_functional_can_be_chosen():

    tags = tags_of(
        ic.incar_text(
            setup_for(
                "Optimization", modules=("GGA",), gga_functional="PS"
            )
        )
    )

    assert tags["GGA"] == "PS"
    assert tags["LASPH"] == ic.TRUE


def test_a_meta_gga_is_named_rather_than_assumed():

    tags = tags_of(
        ic.incar_text(
            setup_for(
                "Optimization",
                modules=("Meta-GGA",),
                meta_gga_functional="RTPSS",
            )
        )
    )

    assert tags["METAGGA"] == "RTPSS"


def test_dispersion_writes_the_scheme_that_was_picked():

    tags = tags_of(
        ic.incar_text(
            setup_for("Optimization", modules=("IVDW",), vdw_ivdw="12")
        )
    )

    assert tags["IVDW"] == "12"


def test_a_hybrid_switches_exact_exchange_on():

    tags = tags_of(
        ic.incar_text(
            setup_for("Optimization", modules=("HSE / Hybrid",))
        )
    )

    assert tags["LHFCALC"] == ic.TRUE
    assert tags["AEXX"] == "0.25"
    assert tags["HFSCREEN"] == "0.2"


def test_pbe0_is_unscreened_and_hse06_is_not():
    """
    The two differ in exactly one number. Carrying HSE06's screening over
    to PBE0 gives HSE06 under another name, and the band gap that comes
    back is not the one that was asked for.
    """

    assert ic.HYBRID_PRESETS["HSE06"] == ("0.25", "0.2")
    assert ic.HYBRID_PRESETS["PBE0"] == ("0.25", "0.0")


def test_solvation_carries_the_permittivity():

    tags = tags_of(
        ic.incar_text(setup_for("Optimization", modules=("VASPsol",)))
    )

    assert tags["LSOL"] == ic.TRUE
    assert tags["EB_K"] == "78.4"


def test_the_optional_blocks_are_written_in_a_fixed_order():
    """
    Two INCARs built from the same choices have to come out identical, or
    every diff is noise.
    """

    forwards = setup_for(
        "Optimization", modules=("GGA", "IVDW", "VASPsol")
    )
    backwards = setup_for(
        "Optimization", modules=("VASPsol", "IVDW", "GGA")
    )

    # Setup takes the order it is given; the dialog sorts it into
    # MODULES order before handing it over, which is what this checks.
    backwards.modules = tuple(
        name for name in ic.MODULES if name in set(backwards.modules)
    )
    forwards.modules = tuple(
        name for name in ic.MODULES if name in set(forwards.modules)
    )

    assert ic.incar_text(forwards) == ic.incar_text(backwards)


def test_every_module_has_a_switch_and_a_description():

    for name in ic.MODULES:
        assert ic.MODULE_NOTES.get(name)


def test_every_module_but_the_hubbard_one_has_settings():
    """
    +U is the exception: its settings depend on which species are open,
    so the dialog builds them rather than reading them from a table.
    """

    for name in ic.MODULES:

        if name == "DFT+U":
            continue

        assert name in ic.MODULE_SECTIONS


# ----------------------------------------------------------------
# Global block
# ----------------------------------------------------------------


def test_the_global_block_reaches_every_job():

    for job in ic.JOBS:

        tags = written(job.name, subjobs=("Static SCF",))

        for tag in ("KPAR", "NCORE", "AMIX", "BMIX"):
            assert tag in tags, job.name


def test_the_global_block_can_turn_the_wavefunction_on():

    assert (
        written("Optimization", global_lwave=ic.TRUE)["LWAVE"] == ic.TRUE
    )


def test_every_global_field_is_written():

    tags = written("Optimization")

    for _, label in ic.GLOBAL_TAGS:
        assert label in tags


# ----------------------------------------------------------------
# Logicals
# ----------------------------------------------------------------


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_every_logical_is_written_the_way_vasp_spells_it(job):
    """
    ``LREAL = False`` is not a logical VASP accepts. Every truth value in
    the file has to be ``.TRUE.`` or ``.FALSE.``, or the run stops on the
    first line it cannot parse.
    """

    tags = written(job, subjobs=("Static SCF",))

    for tag, value in tags.items():

        if value.lower() in {"true", "false", ".t.", ".f."}:
            pytest.fail(f"{job}: {tag} = {value} is not a VASP logical")


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_lreal_is_either_a_logical_or_auto(job):

    value = written(job, subjobs=("Static SCF",))["LREAL"]

    assert value in {ic.TRUE, ic.FALSE, "Auto"}


# ----------------------------------------------------------------
# Field definitions
# ----------------------------------------------------------------


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_every_choice_starts_on_one_of_its_own_options(job):
    """
    A default that is not in the list leaves the dropdown showing the
    first entry while the file carries the default, so what is on screen
    and what is written disagree.
    """

    for field in ic.job_by_name(job).fields():

        if field.kind is not ic.Kind.CHOICE:
            continue

        assert field.default in {
            choice.value for choice in field.choices
        }, f"{job}: {field.name}"


def test_the_optional_and_global_choices_start_on_their_own_options():

    sections = list(ic.MODULE_SECTIONS.values()) + list(ic.GLOBAL_SECTIONS)

    for section in sections:

        for field in section.fields:

            if field.kind is not ic.Kind.CHOICE:
                continue

            assert field.default in {
                choice.value for choice in field.choices
            }, field.name


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_every_numeric_default_sits_inside_its_own_bounds(job):

    for field in ic.job_by_name(job).fields():

        if field.kind not in {ic.Kind.INT, ic.Kind.FLOAT}:
            continue

        value = float(field.default)

        if field.low is not None:
            assert value >= field.low, f"{job}: {field.name}"

        if field.high is not None:
            assert value <= field.high, f"{job}: {field.name}"


@pytest.mark.parametrize("job", [job.name for job in ic.JOBS])
def test_every_setting_says_what_it_does(job):

    for field in ic.job_by_name(job).fields():
        assert field.note, f"{job}: {field.name}"


def test_a_section_with_no_condition_is_always_shown():

    section = ic.Section("Anything", ())

    assert section.visible({})


def test_a_conditional_section_waits_for_its_field():

    section = ic.Section("Thermostat", (), when=(("ensemble", ("NVT",)),))

    assert not section.visible({})
    assert not section.visible({"ensemble": "NPT"})
    assert section.visible({"ensemble": "NVT"})


def test_a_section_with_two_conditions_needs_both():

    section = ic.Section(
        "Nose-Hoover",
        (),
        when=(
            ("ensemble", ("NVT",)),
            ("thermostat", ("Nose-Hoover",)),
        ),
    )

    assert section.visible(
        {"ensemble": "NVT", "thermostat": "Nose-Hoover"}
    )
    assert not section.visible(
        {"ensemble": "NPT", "thermostat": "Nose-Hoover"}
    )
    assert not section.visible({"ensemble": "NVT", "thermostat": "CSVR"})


def test_the_thermostat_panels_belong_to_nvt():
    """
    The thermostat name survives a switch to another ensemble, so a panel
    keyed on it alone follows the user into NPT and offers a setting that
    ensemble never reads.
    """

    npt = {"aimd_ensemble": "NPT", "aimd_nvt_thermostat": "Nose-Hoover"}

    for section in ic.job_by_name("AIMD").sections:

        if "aimd_nvt_thermostat" in section.gates():
            assert not section.visible(npt), section.title


# ----------------------------------------------------------------
# ENCUT
# ----------------------------------------------------------------


def test_the_cutoff_is_raised_above_the_largest_enmax():
    """
    A basis fixed in a moving cell leaves Pulay stress behind, which is
    why VASP asks for 1.3 x ENMAX for anything that relaxes the cell.
    """

    assert ic.encut_for(400.0) == 520


def test_there_is_no_cutoff_to_derive_without_an_enmax():

    assert ic.encut_for(None) is None


def test_the_derived_cutoff_is_clamped_to_something_runnable():

    assert ic.encut_for(10.0) == ic.ENCUT_MIN
    assert ic.encut_for(5000.0) == ic.ENCUT_MAX


def test_every_job_starts_from_the_fallback_cutoff():

    for job in ic.JOBS:

        tags = written(job.name, subjobs=("Static SCF",))

        assert tags["ENCUT"] == str(ic.ENCUT_FALLBACK)
