"""
Tests for POTCAR assembly.

The failures worth guarding against here are the quiet ones. A POTCAR
carries no element names that VASP checks against the POSCAR, so a
mis-ordered or short file is not an error -- it is a run of different
physics that finishes normally. Two real cases drive most of this file:

* the library folder is one level above the elements, because VASP
  distributions unpack into ``potpaw_*`` set directories; and
* an interrupted download leaves the whole directory tree in place with
  empty files in it, which used to resolve as "found".
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from qvista.services.potcar import (
    GW_FAMILY,
    PBE_FAMILY,
    PotcarChoice,
    block_count,
    choice_for,
    concatenate,
    inspect_library,
    library_roots,
    looks_like_library,
    max_enmax,
    ordered_elements,
    read_header,
    recommended_encut,
    resolve_element,
    resolve_structure,
    total_valence,
    variant_candidates,
)


# ------------------------------------------------------------
# A library to test against
# ------------------------------------------------------------


def potcar_text(titel: str, enmax: float, zval: float, lexch: str = "PE") -> str:
    """A minimal but structurally real POTCAR."""

    return (
        f"  {titel}\n"
        f"   {zval:.8f}\n"
        " parameters from PSCTR are:\n"
        f"   LEXCH  = {lexch}\n"
        f"   TITEL  = {titel}\n"
        f"   POMASS =    6.941; ZVAL   =    {zval:.3f}    mass and valenz\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  =  {enmax * 0.75:.3f} eV\n"
        " local part\n"
        "   0.1000 0.2000 0.3000\n"
        " End of Dataset\n"
    )


def write_variant(
    root: Path,
    variant: str,
    enmax: float = 400.0,
    zval: float = 1.0,
    *,
    empty: bool = False,
    gzipped: bool = False,
) -> Path:
    """Create one variant directory in a library."""

    directory = root / variant
    directory.mkdir(parents=True, exist_ok=True)

    text = "" if empty else potcar_text(f"PAW_PBE {variant} 01Jan2000", enmax, zval)

    if gzipped:
        path = directory / "POTCAR.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path = directory / "POTCAR"
        path.write_text(text, encoding="utf-8")

    return path


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A flat library holding Li, Li_sv, Mn_pv, O and a GW variant."""

    root = tmp_path / "potpaw_PBE"

    write_variant(root, "Li", enmax=140.0, zval=1.0)
    write_variant(root, "Li_sv", enmax=499.0, zval=3.0)
    write_variant(root, "Mn_pv", enmax=270.0, zval=13.0)
    write_variant(root, "O", enmax=400.0, zval=6.0)
    write_variant(root, "O_GW", enmax=414.0, zval=6.0)

    return root


# ------------------------------------------------------------
# Preference rules
# ------------------------------------------------------------


def test_pbe_prefers_the_plain_potential(library: Path) -> None:
    """With both present, PBE takes Li over Li_sv."""

    choice = resolve_element(library, "Li", PBE_FAMILY)

    assert choice.variant == "Li"


def test_pbe_falls_back_to_semicore(library: Path) -> None:
    """Mn has only Mn_pv, which the rules reach on the third try."""

    choice = resolve_element(library, "Mn", PBE_FAMILY)

    assert choice.found
    assert choice.variant == "Mn_pv"


def test_gw_never_takes_the_plain_potential(library: Path) -> None:
    """
    GW must not silently fall back to the standard potential.

    The GW variants exist because the standard ones lack the unoccupied
    states; substituting one produces a converged, wrong quasiparticle
    spectrum.
    """

    assert variant_candidates("O", GW_FAMILY) == ["O_GW", "O_d_GW"]

    assert resolve_element(library, "O", GW_FAMILY).variant == "O_GW"

    # Li has no GW variant in this library, and Li on its own is not a
    # substitute for it.
    assert not resolve_element(library, "Li", GW_FAMILY).found


def test_variant_lookup_is_case_insensitive(tmp_path: Path) -> None:
    """Libraries differ on ``Li_GW`` versus ``Li_gw``."""

    root = tmp_path / "lib"
    write_variant(root, "Li_gw")

    assert resolve_element(root, "Li", GW_FAMILY).variant == "Li_gw"


# ------------------------------------------------------------
# Broken and half-downloaded libraries
# ------------------------------------------------------------


def test_empty_potcar_is_not_a_choice(tmp_path: Path) -> None:
    """
    A zero-length POTCAR must not count as found.

    This is the half-downloaded-library case: the tree is complete and
    every file is empty. Concatenating one produces a POTCAR with fewer
    species than the POSCAR, which VASP reads without complaint.
    """

    root = tmp_path / "lib"
    write_variant(root, "Li", empty=True)

    choice = resolve_element(root, "Li", PBE_FAMILY)

    assert not choice.found
    assert choice.path is None
    assert choice.unusable == ("Li",)


def test_empty_variant_falls_through_to_the_next(tmp_path: Path) -> None:
    """A broken first choice must not stop the search."""

    root = tmp_path / "lib"
    write_variant(root, "Li", empty=True)
    write_variant(root, "Li_sv", enmax=499.0, zval=3.0)

    choice = resolve_element(root, "Li", PBE_FAMILY)

    assert choice.variant == "Li_sv"


def test_file_without_a_header_is_rejected(tmp_path: Path) -> None:
    """Non-empty is not enough; it has to parse as a POTCAR."""

    root = tmp_path / "lib"
    directory = root / "Li"
    directory.mkdir(parents=True)
    (directory / "POTCAR").write_text("not a potcar at all\n", encoding="utf-8")

    choice = resolve_element(root, "Li", PBE_FAMILY)

    assert not choice.found
    assert choice.unusable == ("Li",)


def test_unusable_and_absent_read_differently(tmp_path: Path) -> None:
    """
    The status line distinguishes "damaged" from "not covered".

    Same outcome, completely different fix: re-download, or pick another
    variant.
    """

    root = tmp_path / "lib"
    write_variant(root, "Li", empty=True)

    unusable = resolve_element(root, "Li", PBE_FAMILY).status
    absent = resolve_element(root, "Mn", PBE_FAMILY).status

    assert "UNREADABLE" in unusable
    assert "NOT FOUND" in absent


# ------------------------------------------------------------
# Where the library actually is
# ------------------------------------------------------------


def test_flat_library_is_its_own_root(library: Path) -> None:
    """A directory of element directories is already the library."""

    assert looks_like_library(library)
    assert library_roots(library) == [library]


def test_nested_potential_set_is_found(tmp_path: Path) -> None:
    """
    The folder a distribution unpacks into is one level above the elements.

    Pointing the dialog at it used to resolve nothing at all, since the
    only sub-directory is the set, not an element.
    """

    outer = tmp_path / "POTCAR"
    write_variant(outer / "potpaw.52", "Li")

    assert not looks_like_library(outer)
    assert library_roots(outer) == [outer / "potpaw.52"]


def test_every_set_is_offered(tmp_path: Path) -> None:
    """
    Several sets side by side are all returned, none picked.

    PBE and LDA potentials are not interchangeable, so the choice belongs
    to the user rather than to a default.
    """

    outer = tmp_path / "POTCAR"
    write_variant(outer / "potpaw_LDA.54", "Li")
    write_variant(outer / "potpaw_PBE.54", "Li")

    assert [path.name for path in library_roots(outer)] == [
        "potpaw_LDA.54",
        "potpaw_PBE.54",
    ]


def test_stray_files_beside_a_set_do_not_fool_the_search(tmp_path: Path) -> None:
    """
    A set directory keeps loose files next to its element directories.

    The library this was written against has an empty file literally named
    POTCAR sitting in potpaw.52, beside the elements. Matching on the
    filename alone made the folder *above* the set look like the library,
    and every element then resolved as missing.
    """

    outer = tmp_path / "POTCAR"
    inner = outer / "potpaw.52"

    write_variant(inner, "Li")

    (inner / "POTCAR").write_text("", encoding="utf-8")
    (inner / "data_base").write_text("", encoding="utf-8")
    (inner / "potpaw_PBE.52.tar.gz").write_text("", encoding="utf-8")

    assert not looks_like_library(outer)
    assert library_roots(outer) == [inner]


def test_half_downloaded_library_is_still_a_library(tmp_path: Path) -> None:
    """
    Empty files still mark the directory as a library.

    Otherwise the user gets "this is not a library" for a folder that
    plainly is one, instead of "these files are empty".
    """

    outer = tmp_path / "POTCAR"
    write_variant(outer / "potpaw.52", "Li", empty=True)

    assert library_roots(outer) == [outer / "potpaw.52"]


def test_unrelated_directory_is_not_a_library(tmp_path: Path) -> None:
    """Nothing is invented for a folder with no potentials in it."""

    (tmp_path / "notes" / "drafts").mkdir(parents=True)

    assert library_roots(tmp_path / "notes") == []


# ------------------------------------------------------------
# How complete is the library
# ------------------------------------------------------------


def test_report_counts_a_complete_library(library: Path) -> None:
    """Everything present, nothing to warn about."""

    report = inspect_library(library)

    assert report.total == 5
    assert report.empty == 0
    assert report.healthy


def test_report_separates_empty_from_absent(tmp_path: Path) -> None:
    """
    The count is what turns "not working" into something actionable.

    With every variant of every element empty, the dropdowns are full of
    choices and none of them can work -- so the dialog has to say how
    much of the library is missing rather than name the elements again.
    """

    root = tmp_path / "lib"

    write_variant(root, "Li", empty=True)
    write_variant(root, "Li_sv", empty=True)
    write_variant(root, "O", enmax=400.0, zval=6.0)

    report = inspect_library(root)

    assert report.total == 3
    assert report.usable == ("O",)
    assert report.empty == 2
    assert not report.healthy

    assert "1 of 3" in report.summary
    assert "O" in report.summary


def test_report_lists_covered_elements_once(tmp_path: Path) -> None:
    """Variants collapse to their element for the usable-elements list."""

    root = tmp_path / "lib"

    write_variant(root, "Li")
    write_variant(root, "Li_sv")
    write_variant(root, "O")

    assert inspect_library(root).elements == ("Li", "O")


def test_report_ignores_non_element_directories(tmp_path: Path) -> None:
    """A set's data_base folder is not a missing potential."""

    root = tmp_path / "lib"

    write_variant(root, "Li")
    (root / "data_base").mkdir()

    report = inspect_library(root)

    assert report.total == 1
    assert report.healthy


# ------------------------------------------------------------
# Header parsing
# ------------------------------------------------------------


def test_header_fields_are_read(library: Path) -> None:
    """TITEL, LEXCH, ENMAX and ZVAL all come off the header."""

    header = read_header(library / "Li_sv" / "POTCAR")

    assert header is not None
    assert header.titel == "PAW_PBE Li_sv 01Jan2000"
    assert header.functional == "PBE"
    assert header.enmax == pytest.approx(499.0)
    assert header.zval == pytest.approx(3.0)


def test_enmax_stops_at_the_semicolon(library: Path) -> None:
    """
    ENMAX shares its line with ENMIN.

    Reading to the end of the line would take ``266.408; ENMIN ...`` and
    fail to a null cutoff.
    """

    header = read_header(library / "O" / "POTCAR")

    assert header is not None
    assert header.enmax == pytest.approx(400.0)


def test_gzipped_potcar_is_read(tmp_path: Path) -> None:
    """Libraries are commonly left gzipped."""

    root = tmp_path / "lib"
    write_variant(root, "Li", enmax=140.0, zval=1.0, gzipped=True)

    choice = resolve_element(root, "Li", PBE_FAMILY)

    assert choice.found
    assert choice.path.name == "POTCAR.gz"
    assert choice.header.enmax == pytest.approx(140.0)


def test_header_of_a_non_potcar_is_none(tmp_path: Path) -> None:
    """An unparseable file reports as such rather than raising."""

    path = tmp_path / "POTCAR"
    path.write_text("just some text\n", encoding="utf-8")

    assert read_header(path) is None


# ------------------------------------------------------------
# What the potentials imply for the run
# ------------------------------------------------------------


def test_encut_comes_from_the_hardest_potential(library: Path) -> None:
    """ENCUT is set by the largest ENMAX, not by an average."""

    choices = resolve_structure(library, ["Li", "Mn", "O"], PBE_FAMILY)

    assert max_enmax(choices) == pytest.approx(400.0)

    minimum, relax = recommended_encut(choices)

    assert minimum == 400
    assert relax == 520


def test_encut_is_none_without_headers() -> None:
    """Nothing is claimed when nothing was read."""

    assert recommended_encut([PotcarChoice(element="Li")]) is None


def test_valence_counts_the_whole_cell(library: Path) -> None:
    """ZVAL per element times the atom count."""

    choices = resolve_structure(library, ["Li", "O"], PBE_FAMILY)

    # Li 1 electron x 4 atoms, O 6 x 8.
    assert total_valence(choices, {"Li": 4, "O": 8}) == 52


def test_semicore_choice_changes_the_electron_count(library: Path) -> None:
    """
    Overriding Li with Li_sv triples its valence.

    This is the quickest check the user has that the override took
    effect -- 1 electron against 3.
    """

    default = resolve_element(library, "Li", PBE_FAMILY)
    override = choice_for(library, "Li", "Li_sv")

    assert total_valence([default], {"Li": 4}) == 4
    assert total_valence([override], {"Li": 4}) == 12


# ------------------------------------------------------------
# Ordering
# ------------------------------------------------------------


class _Specie:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _Site:
    def __init__(self, symbol: str) -> None:
        self.specie = _Specie(symbol)


def test_elements_keep_poscar_order() -> None:
    """
    Species order is the POTCAR's only link to the POSCAR.

    Sorting them -- which pymatgen's ``symbol_set`` does -- would hand
    every element another element's potential, with no error anywhere.
    """

    structure = [_Site(s) for s in ("O", "O", "Li", "Mn", "Li", "O")]

    assert ordered_elements(structure) == ["O", "Li", "Mn"]


def test_concatenation_follows_that_order(library: Path) -> None:
    """The file is written in the order the choices were resolved."""

    choices = resolve_structure(library, ["O", "Li"], PBE_FAMILY)

    content = concatenate(choices)

    assert content.index("PAW_PBE O ") < content.index("PAW_PBE Li ")


# ------------------------------------------------------------
# Writing it out
# ------------------------------------------------------------


def test_concatenation_holds_one_block_per_element(library: Path) -> None:
    """Three species in, three potentials out."""

    choices = resolve_structure(library, ["Li", "Mn", "O"], PBE_FAMILY)

    assert block_count(concatenate(choices)) == 3


def test_missing_element_refuses_to_write(library: Path) -> None:
    """A gap in the list must not produce a short POTCAR."""

    choices = resolve_structure(library, ["Li", "Fe"], PBE_FAMILY)

    with pytest.raises(ValueError, match="Fe"):
        concatenate(choices)


def test_truncated_potcar_refuses_to_write(tmp_path: Path) -> None:
    """
    A file missing its end marker is not concatenated.

    A part-written POTCAR runs straight into the next element's header,
    and VASP reads the result as one damaged potential.
    """

    root = tmp_path / "lib"
    directory = root / "Li"
    directory.mkdir(parents=True)

    text = potcar_text("PAW_PBE Li 01Jan2000", 140.0, 1.0)
    (directory / "POTCAR").write_text(
        text.replace(" End of Dataset\n", ""), encoding="utf-8"
    )

    choices = resolve_structure(root, ["Li"], PBE_FAMILY)

    with pytest.raises(ValueError, match="truncated"):
        concatenate(choices)


def test_already_concatenated_file_refuses_to_write(tmp_path: Path) -> None:
    """
    A variant directory holding a multi-element POTCAR is rejected.

    It happens when a finished POTCAR is copied back into the library;
    using it would duplicate species.
    """

    root = tmp_path / "lib"
    directory = root / "Li"
    directory.mkdir(parents=True)

    doubled = potcar_text("PAW_PBE Li 01Jan2000", 140.0, 1.0) * 2
    (directory / "POTCAR").write_text(doubled, encoding="utf-8")

    choices = resolve_structure(root, ["Li"], PBE_FAMILY)

    with pytest.raises(ValueError, match="exactly one"):
        concatenate(choices)


def test_every_part_ends_with_a_newline(tmp_path: Path) -> None:
    """
    A part without a trailing newline must not run into the next header.

    Libraries in the wild do ship files like this.
    """

    root = tmp_path / "lib"

    for variant in ("Li", "O"):
        directory = root / variant
        directory.mkdir(parents=True)
        text = potcar_text(f"PAW_PBE {variant} 01Jan2000", 400.0, 1.0)
        (directory / "POTCAR").write_text(text.rstrip("\n"), encoding="utf-8")

    choices = resolve_structure(root, ["Li", "O"], PBE_FAMILY)

    content = concatenate(choices)

    assert "End of Dataset  PAW_PBE" not in content
    assert block_count(content) == 2
