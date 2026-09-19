"""
Menu bar definition and construction.

The menu is described declaratively in ``MENU_STRUCTURE`` and built by
:func:`build_menu_bar`.  Each :class:`Action` names a slot -- a method on the
main window.  Slots that do not exist yet fall back to a placeholder handler,
so the full menu is navigable while individual tools are still being written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Sequence, Union

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenuBar, QWidget


# ------------------------------------------------------------
# Specification types
# ------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """A leaf menu entry bound to a slot on the main window."""

    label: str
    slot: str
    shortcut: str | None = None


@dataclass(frozen=True)
class Separator:
    """A horizontal rule between entries."""


@dataclass(frozen=True)
class Menu:
    """A menu or submenu containing further entries."""

    label: str
    items: Sequence["Entry"] = field(default_factory=tuple)


Entry = Union[Action, Separator, Menu]


# ------------------------------------------------------------
# Menu structure
# ------------------------------------------------------------

MENU_STRUCTURE: tuple[Menu, ...] = (
    Menu(
        "File",
        (
            Action("Open Project", "open_project", "Ctrl+O"),
            Action("Export Data", "export_data", "Ctrl+E"),
            # A structure file says where the atoms are; a figure says
            # what was being looked at. The second is not recoverable
            # from the first, so it gets its own entry beside it.
            Action("Export Image", "export_image", "Ctrl+Shift+E"),
            Separator(),
            # Remembered across runs, so the pseudopotential library is
            # pointed at once rather than browsed for on every structure.
            Action("Set POTCAR Library", "set_potcar_library"),
            # Which CHGNet model the force-field workflows use. Empty
            # means the weights bundled with the installed package;
            # setting it points every one of them at a fine-tuned or
            # alternative model instead.
            Action("Set CHGNet Path", "set_chgnet_path"),
            Separator(),
            Action("Exit", "exit_application", "Ctrl+Q"),
        ),
    ),
    Menu(
        "VASP Input",
        (
            # POTCAR opens the PBE/HSE assembler directly. It used to be a
            # submenu offering GW as well, which put a choice in front of
            # the common case for the sake of the rare one.
            Action("POTCAR", "potcar_pbe_hse"),
            # Likewise one entry rather than a submenu: which of the
            # seven calculations to write is chosen inside the dialog,
            # where each one sits next to a line saying what it is for.
            Action("INCAR", "incar_generator"),
            Menu(
                "K-points",
                (
                    Menu(
                        "Gamma",
                        (
                            Action("K-spacing", "kpoints_gamma_kspacing"),
                            Action("Customised", "kpoints_gamma_customised"),
                        ),
                    ),
                    Menu(
                        "MP",
                        (
                            Action("K-spacing", "kpoints_mp_kspacing"),
                            Action("Customised", "kpoints_mp_customised"),
                        ),
                    ),
                ),
            ),
        ),
    ),
    Menu(
        "Edit Structure",
        (
            # Click an atom in the viewer and it goes. The usual reason
            # is making a vacancy, which is otherwise a matter of
            # counting lines in a POSCAR and deleting the right one.
            Action("Remove Atom", "remove_atom"),
            Separator(),
            # Both of these are file-in, file-out: they work on a path
            # rather than on the open project, so neither needs a
            # structure loaded first.
            Action("Convert", "convert_structure"),
            Action("Selective Dynamics", "selective_dynamics"),
        ),
    ),
    # Moving the atoms downhill, as against measuring them where they
    # are. Everything under Analysis assumes this has been done already:
    # a band structure or a phonon spectrum computed on an unrelaxed cell
    # describes a structure nobody has.
    Menu(
        "Optimization",
        (
            Action("CHGNet", "relax_structure"),
        ),
    ),
    Menu(
        "Analysis",
        (
            Menu(
                "Crystallography",
                (
                    Action("Space Group Analyzer", "space_group_analyzer"),
                    Action("High Symmetry Pathway", "high_symmetry_pathway"),
                ),
            ),
            # Everything here is phonopy, so there is no second thing
            # for a "Phonopy" level to distinguish it from -- it was a
            # submenu of one, and a click to get past. What does differ
            # between the two branches is where the forces come from:
            # CHGNet answers in seconds and is a screening result, VASP
            # is the first-principles answer and is measured in days.
            Menu(
                "Phonon Analysis",
                (
                    # There is no path entry here. A phonon path is a
                    # property of the figure, not of the structure, and
                    # it is chosen on the dispersion itself -- see the
                    # "Select Custom Path" button on the plot. The
                    # Crystallography path builder is a different thing:
                    # it describes the structure and writes KPOINTS.
                    Menu(
                        "CHGNet",
                        (
                            Action("Phonon Band", "phonon_band"),
                            Action(
                                "Band with DOS", "phonon_band_with_dos"
                            ),
                            Action(
                                "Projected Phonon Band",
                                "projected_phonon_band",
                            ),
                            Action(
                                "Band Visualization",
                                "phonon_band_visualization",
                            ),
                            Separator(),
                            # Integrated over the DOS mesh, so this
                            # implies the DOS calculation and is listed
                            # apart from the three dispersion figures.
                            Action(
                                "Thermal Properties", "phonon_thermal"
                            ),
                        ),
                    ),
                    Menu(
                        "VASP",
                        (
                            Action(
                                "Pre-Processing",
                                "vasp_phonon_preprocess",
                            ),
                            # The same four figures as CHGNet above, over
                            # first principles forces. Reading the folder
                            # produces the band, the DOS and the element
                            # projections together, so the four below are
                            # readings of one result: the first one
                            # clicked asks for the folder, and Read
                            # Configuration is how a different folder is
                            # pointed at afterwards.
                            Menu(
                                "Post-Processing",
                                (
                                    Action(
                                        "Read Configuration",
                                        "vasp_phonon_postprocess",
                                    ),
                                    Separator(),
                                    Action(
                                        "Phonon Band", "vasp_phonon_band"
                                    ),
                                    Action(
                                        "Band with DOS",
                                        "vasp_phonon_band_with_dos",
                                    ),
                                    Action(
                                        "Projected Phonon Band",
                                        "vasp_phonon_projected",
                                    ),
                                    Action(
                                        "Band Visualization",
                                        "vasp_phonon_visualization",
                                    ),
                                    Separator(),
                                    Action(
                                        "Thermal Properties",
                                        "vasp_phonon_thermal",
                                    ),
                                ),
                            ),
                        ),
                    ),
                    # There is deliberately no "Visualize Modes" entry
                    # here. Both branches already end in Band
                    # Visualization, which animates the force constants
                    # that branch just produced. A third entry that
                    # animates "whichever were computed last" offers a
                    # choice nobody wants to make and no way to tell
                    # which set is loaded.
                ),
            ),
            Menu(
                "Electrochemical Tools",
                (
                    # One entry each. Both windows already show every
                    # projection as a row of chips, so a submenu listing
                    # the same five options was a second way of saying
                    # what the window says anyway -- and it put a choice
                    # in front of opening the thing that explains it.
                    Action("Band Structure", "band_plot"),
                    Action("DOS", "dos_plot"),
                    # One entry: the window computes every shell it can
                    # and shows them together, so picking one first was
                    # a choice made before there was anything to base
                    # it on.
                    Action("Band Centre", "band_centre"),
                    # The same window at two levels of detail. spd answers
                    # "how much p character"; resolved answers "which d
                    # orbital", which is the difference between t2g and eg
                    # and is invisible once the shell has been summed.
                    Menu(
                        "Orbital",
                        (
                            Action(
                                "Orbital Projected - spd",
                                "orbital_projected",
                            ),
                            Action(
                                "Orbital Resolved - d(5), p(3)",
                                "orbital_resolved",
                            ),
                        ),
                    ),
                ),
            ),
            Menu(
                "Bonding Tools",
                (
                    Action("Bader Analysis", "bader_analysis"),
                    Action("DDEC6", "ddec6_analysis"),
                    # COHP, COOP and COBI all come out of one LOBSTER run
                    # and are read by one window, which picks between them
                    # on a combo box rather than making three menu entries
                    # out of one folder.
                    Action("COHP", "cohp_analysis"),
                ),
            ),
            Menu(
                "Electronegativity",
                (
                    Action("Select Atoms", "electronegativity_panel"),
                    Action(
                        "Polyhedral Environment",
                        "electronegativity_polyhedral",
                    ),
                    Action("Shift Map", "electronegativity_map"),
                    Action("Clear Selection", "electronegativity_clear"),
                ),
            ),
            Separator(),
            # Split the same way Phonopy is: what differs between the two
            # branches is where the forces come from. CHGNet strains the
            # cell here and answers in a minute; VASP is the
            # first-principles answer, computed elsewhere and read back.
            Menu(
                "Mechanical Properties",
                (
                    Action("CHGNet", "mechanical_properties"),
                    Action("VASP", "mechanical_properties_vasp"),
                ),
            ),
            # Same split again: relax the path here with a force field, or
            # write the VASP images out, run them elsewhere, and read the
            # barrier back.
            Menu(
                "NEB",
                (
                    Action("CHGNet", "neb_analysis"),
                    Menu(
                        "VASP",
                        (
                            Action("Pre-Processing", "neb_preprocess"),
                            Action("Post-Processing", "neb_postprocess"),
                        ),
                    ),
                ),
            ),
            # Running the dynamics and reading a trajectory are separate
            # jobs: the second also applies to an XDATCAR from a VASP AIMD
            # run that QVista never launched.
            # Both branches offer the same analyses, because they are
            # the same physics read off a trajectory. What differs is
            # where the trajectory comes from: CHGNet runs it here and
            # the analyses open on it directly, while VASP reads a
            # finished run off disk.
            Menu(
                "MD",
                (
                    Menu(
                        "CHGNet",
                        (
                            Action("Run Dynamics", "md_analysis"),
                            Menu(
                                "Analysis",
                                (
                                    Action(
                                        "MSD and Diffusion",
                                        "chgnet_msd",
                                    ),
                                    Action("RDF", "chgnet_rdf"),
                                    Action(
                                        "Van Hove Correlation",
                                        "chgnet_van_hove",
                                    ),
                                    Action(
                                        "Residence Time",
                                        "chgnet_residence",
                                    ),
                                    Separator(),
                                    Action(
                                        "Diffusion Kinetics (D0, Ea)",
                                        "chgnet_kinetics",
                                    ),
                                ),
                            ),
                        ),
                    ),
                    Menu(
                        "VASP",
                        (
                            # The energies over the run, read from
                            # OSZICAR. Named for what it draws rather
                            # than for launching anything: VASP ran the
                            # dynamics elsewhere, and this reads them.
                            Action("Energy Plot", "md_analysis_vasp"),
                            Menu(
                                "Analysis",
                                (
                                    Action(
                                        "MSD and Diffusion",
                                        "md_diffusion",
                                    ),
                                    Action("RDF", "md_rdf"),
                                    Action(
                                        "Van Hove Correlation",
                                        "md_van_hove",
                                    ),
                                    Action(
                                        "Residence Time", "md_residence"
                                    ),
                                    Separator(),
                                    # Several trajectories, one per
                                    # temperature: an activation energy
                                    # is a slope, not a single run.
                                    Action(
                                        "Diffusion Kinetics (D0, Ea)",
                                        "md_kinetics",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
    Menu(
        "Help",
        (
            Action("About", "show_about"),
        ),
    ),
)


# ------------------------------------------------------------
# Construction
# ------------------------------------------------------------


def build_menu_bar(
    window: QWidget,
    menu_bar: QMenuBar,
    fallback: Callable[[str], None],
) -> dict[str, QAction]:
    """
    Populate ``menu_bar`` from :data:`MENU_STRUCTURE`.

    Actions are connected to the matching method on ``window``.  When that
    method does not exist, the action calls ``fallback`` with its label.

    Returns a mapping of slot name to the created action, so callers can
    enable, disable or retrieve entries later.
    """

    actions: dict[str, QAction] = {}

    for spec in MENU_STRUCTURE:
        menu = menu_bar.addMenu(spec.label)
        _populate(window, menu, spec.items, fallback, actions)

    return actions


def _populate(
    window: QWidget,
    menu,
    items: Sequence[Entry],
    fallback: Callable[[str], None],
    actions: dict[str, QAction],
) -> None:
    """Recursively add ``items`` to ``menu``."""

    for item in items:

        if isinstance(item, Separator):
            menu.addSeparator()
            continue

        if isinstance(item, Menu):
            submenu = menu.addMenu(item.label)
            _populate(window, submenu, item.items, fallback, actions)
            continue

        action = QAction(item.label, window)

        if item.shortcut:
            action.setShortcut(item.shortcut)

        handler = getattr(window, item.slot, None)

        if callable(handler):
            action.triggered.connect(handler)
        else:
            action.triggered.connect(partial(fallback, item.label))

        menu.addAction(action)
        actions[item.slot] = action
