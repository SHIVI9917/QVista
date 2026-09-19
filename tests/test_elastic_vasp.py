"""
Tests for reading elastic constants out of a VASP OUTCAR.

Two things here can be wrong while looking entirely right, and both get
pinned rather than sampled.

The **units**: VASP prints kBar, everything else is GPa, and a factor of
ten on a stiff oxide is not obviously absurd.

The **column order**: VASP writes ``XX YY ZZ XY YZ ZX`` and the Voigt
convention is ``xx yy zz yz xz xy``. The last three are a cyclic rotation
apart, so a tensor read without reordering is still symmetric, still
positive definite, and still passes every Born criterion -- it just has
C44 and C66 swapped. On a cubic crystal the two are equal and nothing
shows. The tests below therefore use a *trigonal* tensor, where the
identity ``C66 = (C11 - C12) / 2`` holds and fails loudly if the
permutation is dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from qvista.services import elastic_vasp as ev


# ------------------------------------------------------------
# Building an OUTCAR to read
# ------------------------------------------------------------


HEADING = {
    "total": "TOTAL ELASTIC MODULI (kBar)",
    "clamped": "SYMMETRIZED ELASTIC MODULI (kBar)",
    "ionic": "ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)",
}


def block_text(name: str, matrix) -> str:
    """One moduli block, laid out the way VASP lays it out."""

    lines = [
        f"  {HEADING[name]}",
        " Direction    XX          YY          ZZ          XY"
        "          YZ          ZX",
        " " + "-" * 80,
    ]

    for label, row in zip(ev.VASP_ORDER, np.asarray(matrix, dtype=float)):
        lines.append(
            f" {label:<8s}" + "".join(f"{value:12.4f}" for value in row)
        )

    lines.append(" " + "-" * 80)

    return "\n".join(lines)


def trigonal_vasp(
    c11=400.0, c12=100.0, c13=50.0, c33=200.0, c44=55.0, c14=6.0
):
    """
    A trigonal tensor written in VASP's column order, in kBar.

    C66 is not free: for this class it is ``(C11 - C12) / 2``, which is
    what makes it a test rather than six arbitrary numbers.
    """

    c66 = (c11 - c12) / 2.0

    tensor = np.zeros((6, 6))

    # VASP order: 0 XX, 1 YY, 2 ZZ, 3 XY, 4 YZ, 5 ZX
    tensor[0, 0] = tensor[1, 1] = c11
    tensor[2, 2] = c33
    tensor[0, 1] = tensor[1, 0] = c12
    tensor[0, 2] = tensor[2, 0] = c13
    tensor[1, 2] = tensor[2, 1] = c13

    # The two shear constants, in VASP's slots.
    tensor[4, 4] = tensor[5, 5] = c44
    tensor[3, 3] = c66

    # The trigonal C14 couples xx to yz, which is VASP's index 4.
    tensor[0, 4] = tensor[4, 0] = c14
    tensor[1, 4] = tensor[4, 1] = -c14
    tensor[3, 5] = tensor[5, 3] = c14

    return tensor


def write_outcar(
    path,
    blocks=None,
    ibrion=6,
    isif=3,
    nfree=4,
    nions=12,
    energy=-123.456,
):
    """Write a minimal OUTCAR carrying the given blocks."""

    if blocks is None:
        clamped = trigonal_vasp()
        ionic = -0.1 * clamped
        blocks = {
            "clamped": clamped,
            "ionic": ionic,
            "total": clamped + ionic,
        }

    parts = [
        f"   IBRION =      {ibrion}    ionic relax: 0-MD 1-quasi-New 2-CG",
        f"   ISIF   =      {isif}  stress and relaxation",
        f"   NFREE  =      {nfree}",
        f"   POTIM  =   0.0150",
        f"   NIONS = {nions}",
        "",
    ]

    for name in ("clamped", "ionic", "total"):
        if name in blocks:
            parts.append(block_text(name, blocks[name]))
            parts.append("")

    parts.append(f"  free  energy   TOTEN  =      {energy:.8f} eV")

    path.write_text("\n".join(parts), encoding="utf-8")

    return path


# ------------------------------------------------------------
# The permutation
# ------------------------------------------------------------


def test_the_permutation_is_not_the_identity() -> None:
    """The whole bug this guards against is doing nothing at all."""

    assert ev.VOIGT_FROM_VASP != (0, 1, 2, 3, 4, 5)


def test_each_voigt_slot_takes_the_right_vasp_column() -> None:
    """
    Pinned one by one, because a plausible-looking permutation that is
    wrong produces a plausible-looking tensor.
    """

    # A matrix whose entries name their own position.
    marked = np.arange(36, dtype=float).reshape(6, 6)

    moved = ev.to_voigt(marked)

    for voigt, vasp in enumerate(ev.VOIGT_FROM_VASP):
        for other_voigt, other_vasp in enumerate(ev.VOIGT_FROM_VASP):

            assert moved[voigt, other_voigt] == marked[vasp, other_vasp]


def test_yz_and_xz_are_not_swapped() -> None:
    """
    Voigt 4 is yz and Voigt 5 is xz; VASP lists YZ before ZX.

    Reversing the shear block rather than rotating it -- ``(0,1,2,5,4,3)``
    -- also leaves the tensor symmetric, also preserves the eigenvalues,
    and also satisfies ``C66 = (C11 - C12) / 2``, because it agrees with
    the right answer about where xy goes. It differs only in swapping yz
    with xz, which is invisible on any crystal where C44 = C55.

    An off-diagonal coupling is what separates them: C14 is the xx-yz
    entry, so it must land in column 3 and nowhere else.
    """

    tensor = np.zeros((6, 6))

    # VASP index 4 is YZ; the XX-YZ coupling is what Voigt calls C14.
    tensor[0, 4] = tensor[4, 0] = 1.0

    moved = ev.to_voigt(tensor)

    assert moved[0, 3] == 1.0, "C14 should be the xx-yz coupling"
    assert moved[0, 4] == 0.0, "yz and xz have been swapped"

    # And xy still lands where both candidate permutations agree it does.
    marked = np.zeros((6, 6))
    marked[3, 3] = 1.0

    assert ev.to_voigt(marked)[5, 5] == 1.0


def test_reordering_preserves_symmetry_and_eigenvalues() -> None:
    """A relabelling cannot change the physics it labels."""

    tensor = trigonal_vasp()

    moved = ev.to_voigt(tensor)

    assert np.allclose(moved, moved.T)

    assert np.allclose(
        np.sort(np.linalg.eigvalsh(tensor)),
        np.sort(np.linalg.eigvalsh(moved)),
    )


def test_a_block_of_the_wrong_shape_is_refused() -> None:

    with pytest.raises(ev.ElasticVaspError, match="6x6"):
        ev.to_voigt(np.zeros((3, 3)))


# ------------------------------------------------------------
# Reading
# ------------------------------------------------------------


def test_all_three_blocks_are_found(tmp_path) -> None:

    result = ev.read_outcar(write_outcar(tmp_path / "OUTCAR"))

    assert set(result.available()) == {"total", "clamped", "ionic"}


def test_a_folder_works_as_well_as_the_file(tmp_path) -> None:

    write_outcar(tmp_path / "OUTCAR")

    assert ev.read_outcar(tmp_path).total is not None


def test_kbar_becomes_gpa(tmp_path) -> None:
    """The factor of ten that is invisible when it is missing."""

    clamped = trigonal_vasp(c11=4000.0)

    write_outcar(
        tmp_path / "OUTCAR",
        blocks={"total": clamped},
    )

    result = ev.read_outcar(tmp_path)

    # 4000 kBar is 400 GPa, not 4000.
    assert result.total[0, 0] == pytest.approx(400.0)


def test_the_trigonal_identity_survives_the_round_trip(tmp_path) -> None:
    """
    ``C66 = (C11 - C12) / 2`` for a trigonal crystal.

    This is the test that fails if the reordering is dropped: without
    it C66 comes back as the shear constant that belongs at C44, and
    the identity is out by a factor of several.
    """

    write_outcar(
        tmp_path / "OUTCAR",
        blocks={"total": trigonal_vasp(c11=400.0, c12=100.0)},
    )

    tensor = ev.read_outcar(tmp_path).total

    assert tensor[5, 5] == pytest.approx(
        (tensor[0, 0] - tensor[0, 1]) / 2.0
    )

    # And the other shear constant is genuinely different, so the
    # identity above was not satisfied by everything being equal.
    assert tensor[3, 3] != pytest.approx(tensor[5, 5])


def test_the_total_is_the_sum_of_the_other_two(tmp_path) -> None:
    """VASP computes it that way, so it is a free check on the parse."""

    result = ev.read_outcar(write_outcar(tmp_path / "OUTCAR"))

    assert result.sum_error < ev.SUM_TOLERANCE


def test_a_mismatched_total_is_reported(tmp_path) -> None:

    clamped = trigonal_vasp()

    write_outcar(
        tmp_path / "OUTCAR",
        blocks={
            "clamped": clamped,
            "ionic": -0.1 * clamped,
            # Not the sum of the two above.
            "total": clamped,
        },
    )

    result = ev.read_outcar(tmp_path)

    assert result.sum_error > ev.SUM_TOLERANCE

    assert any("not the sum" in note for note in result.notes)


def test_the_last_block_wins(tmp_path) -> None:
    """VASP prints the moduli once per ionic step in some setups."""

    first = trigonal_vasp(c11=100.0)
    second = trigonal_vasp(c11=900.0)

    text = (
        block_text("total", first)
        + "\n\n"
        + block_text("total", second)
    )

    (tmp_path / "OUTCAR").write_text(
        "   IBRION =      6\n   ISIF   =      3\n" + text, encoding="utf-8"
    )

    result = ev.read_outcar(tmp_path)

    assert result.total[0, 0] == pytest.approx(90.0)


def test_the_tags_come_back(tmp_path) -> None:

    result = ev.read_outcar(
        write_outcar(tmp_path / "OUTCAR", ibrion=6, isif=3, nions=48)
    )

    assert result.ibrion == 6
    assert result.isif == 3
    assert result.natoms == 48
    assert result.energy == pytest.approx(-123.456)


# ------------------------------------------------------------
# When the run was the wrong kind
# ------------------------------------------------------------


def test_a_missing_outcar_says_what_to_do(tmp_path) -> None:

    with pytest.raises(ev.ElasticVaspError, match="No OUTCAR"):
        ev.read_outcar(tmp_path)


def test_a_missing_folder_is_refused(tmp_path) -> None:

    with pytest.raises(ev.ElasticVaspError, match="no such folder"):
        ev.read_outcar(tmp_path / "nowhere")


def test_an_outcar_with_no_moduli_names_ibrion(tmp_path) -> None:

    (tmp_path / "OUTCAR").write_text(
        "   IBRION =      2\n   ISIF   =      3\n", encoding="utf-8"
    )

    with pytest.raises(ev.ElasticVaspError, match="IBRION"):
        ev.read_outcar(tmp_path)


def test_the_wrong_ibrion_is_flagged(tmp_path) -> None:
    """The blocks are still read; the run is still called out."""

    result = ev.read_outcar(
        write_outcar(tmp_path / "OUTCAR", ibrion=8)
    )

    assert any("IBRION = 8" in note for note in result.notes)


def test_a_fixed_cell_is_flagged(tmp_path) -> None:

    result = ev.read_outcar(
        write_outcar(tmp_path / "OUTCAR", isif=2)
    )

    assert any("ISIF = 2" in note for note in result.notes)


def test_a_clamped_only_run_is_flagged(tmp_path) -> None:

    write_outcar(
        tmp_path / "OUTCAR",
        blocks={"clamped": trigonal_vasp()},
    )

    result = ev.read_outcar(tmp_path)

    assert "total" not in result.tensors

    assert any("no TOTAL" in note for note in result.notes)


# ------------------------------------------------------------
# Derived quantities
# ------------------------------------------------------------


def test_the_ionic_contribution_is_measured(tmp_path) -> None:
    """A tenth of the stiffness removed is a tenth of the bulk modulus."""

    clamped = trigonal_vasp()

    write_outcar(
        tmp_path / "OUTCAR",
        blocks={
            "clamped": clamped,
            "ionic": -0.1 * clamped,
            "total": 0.9 * clamped,
        },
    )

    result = ev.read_outcar(tmp_path)

    assert result.ionic_fraction == pytest.approx(0.1, abs=1e-6)


def test_a_large_ionic_contribution_is_called_out(tmp_path) -> None:

    clamped = trigonal_vasp()

    write_outcar(
        tmp_path / "OUTCAR",
        blocks={
            "clamped": clamped,
            "ionic": -0.4 * clamped,
            "total": 0.6 * clamped,
        },
    )

    result = ev.read_outcar(tmp_path)

    assert any("Ionic relaxation" in note for note in result.notes)


def test_the_ionic_fraction_is_undefined_without_both_blocks(
    tmp_path,
) -> None:

    write_outcar(
        tmp_path / "OUTCAR", blocks={"total": trigonal_vasp()}
    )

    assert np.isnan(ev.read_outcar(tmp_path).ionic_fraction)


# ------------------------------------------------------------
# Into the shared result
# ------------------------------------------------------------


def test_the_result_carries_moduli_and_stability(tmp_path) -> None:

    result = ev.to_result(ev.read_outcar(write_outcar(tmp_path / "OUTCAR")))

    assert result.moduli is not None
    assert result.stability is not None

    assert result.stability.stable

    assert result.moduli.bulk_hill > 0.0


def test_the_result_says_it_came_from_vasp(tmp_path) -> None:
    """
    So a screenshot can never be attributed to the wrong method.

    The panel prints this line verbatim under the tensor.
    """

    result = ev.to_result(ev.read_outcar(write_outcar(tmp_path / "OUTCAR")))

    assert "VASP" in result.force_field
    assert "IBRION = 6" in result.force_field


def test_the_moduli_match_the_force_field_route(tmp_path) -> None:
    """
    Both routes must derive moduli identically from the same tensor.

    The point of sharing :func:`~qvista.analyses.elastic.derive_moduli`
    is that a VASP tensor and a CHGNet tensor are comparable; if the two
    paths ever diverged the comparison would be meaningless.
    """

    from qvista.analyses.elastic import derive_moduli

    elastic = ev.read_outcar(write_outcar(tmp_path / "OUTCAR"))

    direct = derive_moduli(elastic.total)

    through = ev.to_result(elastic).moduli

    assert through.bulk_hill == pytest.approx(direct.bulk_hill)
    assert through.shear_hill == pytest.approx(direct.shear_hill)
    assert through.youngs == pytest.approx(direct.youngs)


def test_a_block_the_run_lacks_is_refused(tmp_path) -> None:

    write_outcar(
        tmp_path / "OUTCAR", blocks={"total": trigonal_vasp()}
    )

    with pytest.raises(ev.ElasticVaspError, match="no SYMMETRIZED"):
        ev.to_result(ev.read_outcar(tmp_path), which="clamped")


def test_an_unknown_block_is_refused(tmp_path) -> None:

    with pytest.raises(ev.ElasticVaspError, match="Unknown block"):
        ev.to_result(
            ev.read_outcar(write_outcar(tmp_path / "OUTCAR")),
            which="springs",
        )


def test_the_clamped_tensor_is_stiffer_than_the_total(tmp_path) -> None:
    """
    Letting the ions relax can only soften a crystal.

    A sign error on the ionic block would reverse this and still produce
    a stable, symmetric, plausible tensor.
    """

    elastic = ev.read_outcar(write_outcar(tmp_path / "OUTCAR"))

    total = ev.to_result(elastic, which="total").moduli
    clamped = ev.to_result(elastic, which="clamped").moduli

    assert clamped.bulk_hill > total.bulk_hill


# ------------------------------------------------------------
# Naming the crystal
# ------------------------------------------------------------


def write_poscar(path) -> None:
    """A simple cubic cell, so the space group is unambiguous."""

    (path / "CONTCAR").write_text(
        "\n".join(
            [
                "test",
                "   1.0",
                "  4.0 0.0 0.0",
                "  0.0 4.0 0.0",
                "  0.0 0.0 4.0",
                "   Po",
                "    1",
                "Direct",
                "  0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_the_space_group_is_read_from_the_folder(tmp_path) -> None:
    """
    CONTCAR sits beside OUTCAR, so nothing needs to be asked for.

    Without it the panel prints an empty symbol and "#0", which reads
    as a failed calculation rather than as a missing label.
    """

    write_outcar(tmp_path / "OUTCAR")
    write_poscar(tmp_path)

    result = ev.to_result(ev.read_outcar(tmp_path))

    assert result.spacegroup_number_after == 221
    assert result.spacegroup_after == "Pm-3m"


def test_a_folder_with_no_structure_is_still_readable(tmp_path) -> None:
    """The tensor does not depend on it; only the label does."""

    result = ev.to_result(ev.read_outcar(write_outcar(tmp_path / "OUTCAR")))

    assert result.spacegroup_number_after == 0
    assert result.tensor is not None


def test_contcar_is_preferred_over_poscar(tmp_path) -> None:
    """
    An ISIF = 3 run moved away from its POSCAR before measuring.

    The moduli describe the relaxed cell, so that is the one to name.
    """

    write_outcar(tmp_path / "OUTCAR")
    write_poscar(tmp_path)

    # A POSCAR that would give a different answer if it were preferred.
    (tmp_path / "POSCAR").write_text(
        "\n".join(
            [
                "decoy",
                "   1.0",
                "  4.0 0.0 0.0",
                "  0.0 5.0 0.0",
                "  0.0 0.0 6.0",
                "   Po",
                "    1",
                "Direct",
                "  0.0 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = ev.to_result(ev.read_outcar(tmp_path))

    # Cubic, from CONTCAR -- not the orthorhombic decoy.
    assert result.spacegroup_number_after == 221
