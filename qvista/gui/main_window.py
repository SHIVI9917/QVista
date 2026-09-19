"""
Main application window for QVista.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from qvista.core.config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)


from qvista.models.project import Project
from qvista.controllers.project_controller import ProjectController

from qvista.analyses.base import Analysis, AnalysisError, AnalysisResult
from qvista.analyses.guards import (
    detect_vacuum,
    require_periodic,
    symmetry_preview,
)
from qvista.analyses.elastic import ElasticAnalysis, ElasticResult
from qvista.analyses.kpath import KPathAnalysis, KPathResult
from qvista.analyses.kpoints import GAMMA_SCHEME, MP_SCHEME
from qvista.analyses.md import MDAnalysis, MDResult
from qvista.analyses.neb import NEBAnalysis, NEBResult
from qvista.analyses.spacegroup import SpaceGroupAnalysis, SpaceGroupResult
from qvista.analyses.phonon import (
    MODE_BAND,
    MODE_BOTH,
    MODE_DOS,
    MODE_PROJECTED,
    MODE_THERMAL,
    PhononAnalysis,
    PhononData,
    PhononResult,
    count_displacements,
    suggest_supercell,
)
from qvista.analyses.worker import AnalysisWorker

from qvista.services import forcefield, phonopy_vasp
from qvista.services import orbital as ob_service
from qvista.services.structure_analyzer import DEFAULT_SYMPREC
from qvista.services.potcar import (
    GW_FAMILY,
    PBE_FAMILY,
    ordered_elements,
    species_counts,
)
from qvista.services.structure_writer import (
    StructureWriteError,
    convert,
    formats_for,
)

from qvista.gui.dialogs.kpath_dialog import (
    KPathDialog,
    SymmetryPreviewDialog,
)
from qvista.gui.dialogs.kpoints_dialog import (
    CustomisedKpointsDialog,
    KSpacingDialog,
)
from qvista.gui.dialogs.potcar_dialog import PotcarDialog
from qvista.gui.dialogs.electronegativity_dialog import (
    ElectronegativityDialog,
)
from qvista.gui.dialogs.dynamics_dialog import MDDialog, NEBDialog
from qvista.gui.dialogs.supercell_dialog import SupercellDialog

from qvista.gui.menu_bar import build_menu_bar

from qvista.gui.widgets.console import Console
from qvista.gui.dialogs.elastic_window import ElasticWindow
from qvista.gui.dialogs.mode_dialog import ModeAnimationDialog
from qvista.gui.dialogs.plot_window import PlotWindow
from qvista.gui.widgets.visualization import VisualizationPanel
from qvista.gui.widgets.workspace import Workspace

#: Result lines kept when Space Group Analyzer is run by hand -- enough for
#: the space group, its number, the Hall symbol, the point group and the
#: crystal system. The full report already printed itself on load.
SUMMARY_LINES = 5



def _path_label(segments) -> str:
    """
    A k-path written the way it is printed under a dispersion.

    Consecutive segments join with a dash; a break between them -- where
    the path jumps rather than continuing -- becomes a vertical bar, the
    same mark the figure's axis uses.
    """

    if not segments:
        return "(empty)"

    parts: list[str] = []

    previous = None

    for start, end in segments:

        if previous is None:
            parts.append(start)

        elif start != previous:
            parts.append(f"|{start}")

        parts.append(f"-{end}")

        previous = end

    return "".join(parts).replace("-|", " | ")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.project_controller = ProjectController()

        #: The analysis currently running, if any. Held as an attribute
        #: because a QThread that goes out of scope is destroyed mid-run.
        self._analysis_worker: AnalysisWorker | None = None

        #: Callback given to :meth:`_run_analysis` for the current run.
        self._analysis_on_success = None

        #: Console verbosity of the current run. See :meth:`_run_analysis`.
        self._analysis_verbose = True
        self._analysis_max_result_lines: int | None = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)

        self._create_central_widget()
        self._create_docks()
        self._create_menu()

        # Connect all GUI signals
        self._connect_signals()



        self.statusBar().showMessage("Ready")
    # ------------------------------------------------------------
    # Central Widget
    # ------------------------------------------------------------

    def _create_central_widget(self) -> None:
        """Create the workspace."""

        self.workspace = Workspace()
        self.setCentralWidget(self.workspace)

    # ------------------------------------------------------------
    # Dock Widgets
    # ------------------------------------------------------------

    def _create_docks(self) -> None:

        # ------------------------------------------------------------
        # Create widgets
        # ------------------------------------------------------------

        self.visualization = VisualizationPanel()

        self.console = Console()

        # Plots live in their own window rather than a dock: figures want
        # roughly square space, which a dock tabbed with the console
        # cannot give them.
        self.plot_window = PlotWindow(self)

        # The elastic tensor is a table, so it gets its own window rather
        # than a dock: the bottom strip belongs to the console.
        self.elastic_window = ElasticWindow(self)

        # ------------------------------------------------------------
        # Create docks
        # ------------------------------------------------------------

        visualization_dock = QDockWidget("Visualization", self)
        visualization_dock.setWidget(self.visualization)

        # ------------------------------------------------------------
        # The left column is pinned too.
        #
        # Every panel keeps a fixed place: none can be dragged out,
        # floated, closed or re-tabbed. The splitters between them still
        # work, so sizes stay adjustable while the layout itself does not
        # move.
        # ------------------------------------------------------------

        visualization_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        visualization_dock.setAllowedAreas(Qt.LeftDockWidgetArea)

        visualization_dock.setMinimumWidth(240)

        # ------------------------------------------------------------
        # Console: pinned to the bottom.
        #
        # No title bar features, so it cannot be dragged out, floated or
        # closed, and only the bottom area will accept it. It is the one
        # output surface, so it should always be exactly where it was.
        # ------------------------------------------------------------

        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setWidget(self.console)

        self.console_dock.setFeatures(
            QDockWidget.NoDockWidgetFeatures
        )

        self.console_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        self.console_dock.setMinimumHeight(180)

        # ------------------------------------------------------------
        # Add docks
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Give the bottom-left corner to the left column.
        #
        # By default the bottom dock area owns both bottom corners, so the
        # console runs the full window width and the left column is cut
        # short above it. Handing the corner to the left area instead lets
        # Visualization run the full height, and the console starts just
        # right of it.
        # ------------------------------------------------------------

        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)

        self.addDockWidget(Qt.LeftDockWidgetArea, visualization_dock)

        # The console owns the bottom strip alone: nothing is tabbed with
        # it, so it never gets hidden behind another panel.
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)

    # ------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------

    def _create_menu(self) -> None:
        """Build the menu bar from the declarative structure."""

        self.menu_actions = build_menu_bar(
            self,
            self.menuBar(),
            self._not_implemented,
        )

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------

    def open_project(self) -> None:
        """Open a VASP project."""

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            (
                "VASP Files (POSCAR CONTCAR vasprun.xml);;"
                "CIF Files (*.cif);;"
                "XYZ Files (*.xyz);;"
                "All Files (*)"
            ),
        )

        if not filename:
            return

        project = self.project_controller.open_project(filename)
        self.update_project_ui(project)

        self.statusBar().showMessage(
            f"Loaded project: {project.info.name}"
        )

    def export_image(self) -> None:
        """
        Save the viewport as an image.

        Deliberately independent of whether a project is open in the
        result sense: what is being exported is the picture, and the
        picture is whatever the viewer is showing. It does insist on a
        structure, because otherwise the export is a photograph of the
        "no structure loaded" hint.
        """

        from qvista.gui.dialogs.image_dialog import ImageDialog

        viewer = self.workspace.viewer

        if viewer.structure is None:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Open a structure first -- there is no picture yet.",
            )
            return

        # Name the file after the structure, so a folder of exports is
        # not a row of "structure.png".
        suggested = "structure"

        project = self.project_controller.current_project()

        if project is not None:

            source = project.info.source_file

            if source:
                suggested = Path(source).stem

        ImageDialog(viewer, suggested, parent=self).exec()

    def export_data(self) -> None:
        """
        Export computed results to disk.

        Nothing is written automatically during a calculation; saving is
        always an explicit action taken here.
        """

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return

        # ----------------------------------------------------------
        # What is available to export?
        # ----------------------------------------------------------

        elastic = project.results.elastic.get("result")
        kpath = project.results.kpath.get("result")

        #: ``(label, default filename, file filter, writer(path))``
        choices: list[tuple[str, str, str, callable]] = []

        # ----------------------------------------------------------
        # Structure conversion.
        #
        # Output format is independent of input format: any of these can
        # be written whatever the file was opened from.
        # ----------------------------------------------------------

        source = project.info.source_file

        info = project.structure_info

        if source is not None and info is not None:

            for fmt in formats_for(info.periodic):

                choices.append(
                    (
                        f"Structure as {fmt.label}",
                        fmt.default_name,
                        fmt.file_filter,
                        partial(convert, source, fmt=fmt.key),
                    )
                )

        # ----------------------------------------------------------
        # Computed results
        # ----------------------------------------------------------

        if isinstance(kpath, KPathResult):
            choices.append(
                (
                    "k-path as VASP KPOINTS",
                    "KPOINTS",
                    "KPOINTS File (KPOINTS);;All Files (*)",
                    lambda path: Path(path).write_text(
                        kpath.to_kpoints(), encoding="utf-8"
                    ),
                )
            )

        if isinstance(elastic, ElasticResult):
            choices.append(
                (
                    "Elastic tensor as CSV",
                    "elastic_tensor.csv",
                    "CSV File (*.csv);;All Files (*)",
                    lambda path: Path(path).write_text(
                        elastic.to_csv(), encoding="utf-8"
                    ),
                )
            )

        if not choices:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Open a structure file first.\n\n"
                "Once a structure is loaded it can be saved as POSCAR, "
                "CIF or XYZ, whatever format it came from.",
            )
            return

        # ----------------------------------------------------------
        # Pick one when several are available
        # ----------------------------------------------------------

        if len(choices) == 1:
            label, default_name, filters, writer = choices[0]

        else:
            labels = [choice[0] for choice in choices]

            chosen, accepted = QInputDialog.getItem(
                self,
                "Export Data",
                "What would you like to export?",
                labels,
                0,
                False,
            )

            if not accepted:
                return

            label, default_name, filters, writer = choices[
                labels.index(chosen)
            ]

        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {label}",
            default_name,
            filters,
        )

        if not filename:
            return

        try:
            writer(filename)

        except StructureWriteError as exc:
            QMessageBox.warning(self, "Cannot convert", str(exc))
            return

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not write {filename}: {exc}",
            )
            return

        self.log(f"Exported {label} to {filename}")

        self.statusBar().showMessage(f"Exported to {filename}", 5000)

    def exit_application(self) -> None:
        """Quit QVista."""

        QApplication.quit()

    def _not_implemented(self, label: str, *_) -> None:
        """
        Placeholder for menu entries whose tool is not written yet.

        Extra arguments are ignored so this can be connected directly to
        ``QAction.triggered``, which emits a checked flag.
        """

        self.log(f"[not implemented] {label}")

        self.statusBar().showMessage(
            f"{label} is not implemented yet",
            4000,
        )

    def log(self, message: str) -> None:
        """
        Write a message to the console.
        """

        self.console.append(message)



    def update_project_ui(self, project: Project) -> None:
        """
        Refresh the entire GUI after loading a project.
        """

    # Console -- the symmetry report, and nothing else. The
    # project/file/structure summary that used to print here is gone.
        self.console.clear()
        self._auto_space_group_analysis(project)
    # Crystal Viewer
        if project.structure is not None:
            self.workspace.load_structure(project.structure)

    def _auto_space_group_analysis(self, project: Project) -> None:
        """
        Run the space group analysis as soon as a project is opened.

        Runs quietly: the heading, the progress chatter and the closing
        "finished" line are all suppressed, so the console opens straight
        onto the results table and everything below it.
        """

        try:
            require_periodic(project)

        except AnalysisError:
            # No lattice to analyse (a plain XYZ, or nothing loaded) --
            # fall back to the summary, which is all there is to show.
            self.update_console(project)
            return

        self._run_analysis(
            SpaceGroupAnalysis(project),
            on_success=self._store_spacegroup,
            verbose=False,
        )


    def update_console(self, project: Project) -> None:
        """
        Update the console with project information.
        """

        self.console.clear()

        self.log(f"Project : {project.info.name}")
        self.log(f"Directory: {project.info.directory}")
        self.log("")

        self.log("Detected files")
        self.log("----------------")

        for filename in sorted(project.info.files):
            self.log(f"✓ {filename}")

        if project.info.missing:
            self.log("")
            self.log("Missing files")
            self.log("----------------")

            for filename in sorted(project.info.missing):
                self.log(f"✗ {filename}")

        self.update_structure_properties(project)

    def update_structure_properties(self, project: Project) -> None:
        """
        Write the structural properties of the loaded structure.
        """

        self.log("")
        self.log("Structure properties")
        self.log("----------------")

        info = project.structure_info

        if info is None:
            self.log(
                project.structure_error
                or "No structure file found in this project."
            )
            return

        if project.info.source_file is not None:
            self.log(
                f"Source : {project.info.source_file.name}"
                f"  [{info.source_format}]"
            )
            self.log("")

        for label, value in info.as_rows():
            self.log(f"{label:<19}: {value}")

        if not info.periodic:
            self.log("")
            self.log(
                "No lattice in this file - space group, crystal system "
                "and volume are undefined."
            )

    # ------------------------------------------------------------
    # Crystallography : Space Group Analyzer
    # ------------------------------------------------------------

    def space_group_analyzer(self) -> None:
        """Report the space group, point group and Wyckoff positions."""

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return

        try:
            require_periodic(project)

        except AnalysisError as exc:
            QMessageBox.warning(self, "Space Group Analyzer", str(exc))
            return

        analysis = SpaceGroupAnalysis(project)

        try:
            structure = analysis.structure()

        except AnalysisError as exc:
            QMessageBox.critical(self, "Space Group Analyzer", str(exc))
            return

        # The answer depends on the tolerance, so let the user see it and
        # adjust before committing.
        dialog = SymmetryPreviewDialog(
            lambda symprec: symmetry_preview(structure, symprec),
            analysis.symprec,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.statusBar().showMessage("Cancelled", 4000)
            return

        analysis.symprec = dialog.symprec

        # The full report already printed itself when the project was
        # opened. Run by hand, this is the short answer: the console is
        # cleared and only the headline rows are kept.
        self.console.clear()

        self._run_analysis(
            analysis,
            on_success=self._store_spacegroup,
            verbose=False,
            max_result_lines=SUMMARY_LINES,
        )

    def _store_spacegroup(self, result: object) -> None:
        """Keep the symmetry description on the project."""

        project = self.project_controller.current_project()

        if project is None or not isinstance(result, SpaceGroupResult):
            return

        project.results.spacegroup = result.as_storage()

    # ------------------------------------------------------------
    # VASP Input
    # ------------------------------------------------------------

    def _prepare_vasp_structure(
        self,
        title: str,
        require_lattice: bool = True,
    ):
        """
        Return the current project's structure, or ``None``.

        VASP input generation needs no worker thread and no symmetry
        tolerance, so this is a much smaller version of
        :meth:`_prepare_kpath`. ``require_lattice`` is False for POTCAR,
        which only needs the element list.
        """

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return None

        if require_lattice:

            try:
                require_periodic(project)

            except AnalysisError as exc:
                QMessageBox.warning(self, title, str(exc))
                return None

        from qvista.services.structure_analyzer import (
            StructureAnalysisError,
            StructureAnalyzer,
        )

        try:
            parsed = StructureAnalyzer.parse(project.info.source_file)

        except StructureAnalysisError as exc:
            QMessageBox.critical(self, title, str(exc))
            return None

        return parsed.structure

    # ------------------------------------------------------------
    # VASP Input : POTCAR
    # ------------------------------------------------------------

    def _potcar_dialog(self, family: str) -> None:
        """Open the POTCAR assembler for one potential family."""

        structure = self._prepare_vasp_structure(
            "POTCAR", require_lattice=False
        )

        if structure is None:
            return

        elements = ordered_elements(structure)

        if not elements:
            QMessageBox.warning(
                self, "POTCAR", "This structure has no atoms."
            )
            return

        # Atoms per element, so the dialog can turn the potentials' ZVAL
        # into the cell's electron count.
        counts = {
            element: sum(
                1
                for site in structure
                if site.specie.symbol == element
            )
            for element in elements
        }

        PotcarDialog(elements, family, counts=counts, parent=self).exec()

    def set_potcar_library(self) -> None:
        """
        Point QVista at the pseudopotential library, once and for all.

        The POTCAR dialog remembers whatever was last browsed to, but
        that only happens after a structure is open and the assembler has
        been reached. Setting it here is independent of any project, so a
        new machine is configured once rather than on the first POTCAR of
        every session.
        """

        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QInputDialog

        from qvista.core.config import APP_NAME, ORGANIZATION
        from qvista.gui.dialogs.potcar_dialog import (
            LIBRARY_KEY,
            POTENTIAL_SET_KEY,
        )
        from qvista.services.potcar import inspect_library, library_roots

        settings = QSettings(ORGANIZATION, APP_NAME)

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select the VASP pseudopotential library",
            settings.value(LIBRARY_KEY, "", type=str),
        )

        if not directory:
            return

        root = Path(directory)

        sets = library_roots(root)

        if not sets:
            QMessageBox.warning(
                self,
                "POTCAR library",
                f"No potentials under {root}.\n\n"
                f"Expected element directories such as Li/POTCAR, either "
                f"there or one level down in a potpaw_* set.",
            )
            return

        chosen = sets[0]

        # Several sets side by side are not interchangeable -- PBE and
        # LDA are different physics -- so the choice is the user's.
        if len(sets) > 1:

            name, accepted = QInputDialog.getItem(
                self,
                "POTCAR library",
                "This folder holds more than one potential set.\n"
                "Which one should QVista use?",
                [candidate.name for candidate in sets],
                0,
                False,
            )

            if not accepted:
                return

            chosen = next(c for c in sets if c.name == name)

        settings.setValue(LIBRARY_KEY, str(root))
        settings.setValue(POTENTIAL_SET_KEY, chosen.name)

        report = inspect_library(chosen)

        self.log(f"POTCAR library set to {chosen}")
        self.log(f"  {report.summary}")

        QMessageBox.information(
            self,
            "POTCAR library",
            f"QVista will use:\n{chosen}\n\n{report.summary}\n\n"
            f"This is remembered between runs.",
        )

    def set_chgnet_path(self) -> None:
        """
        Choose which CHGNet model every force-field workflow uses.

        One setting rather than one per analysis: relaxation, phonons,
        NEB, molecular dynamics and elastic constants all reach the
        force field through the same place, so pointing that at a
        different model points all of them at it. A study that used a
        fine-tuned model for the NEB and the stock one for the phonons
        would be comparing two potentials without saying so.

        Empty means the weights bundled with the installed package,
        which is what it is until someone sets it.
        """

        from PySide6.QtCore import QSettings

        from qvista.core.config import APP_NAME, ORGANIZATION
        from qvista.services import forcefield

        settings = QSettings(ORGANIZATION, APP_NAME)

        current = forcefield.chgnet_model_path()

        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Select a CHGNet model file",
            current or "",
            "CHGNet model (*.pth.tar *.pt *.pth);;All files (*)",
        )

        if not chosen:

            # Cancelling with something already set is ambiguous, so ask
            # rather than guess: the two possible intentions are "leave
            # it alone" and "go back to the bundled model".
            if current and QMessageBox.question(
                self,
                "CHGNet model",
                f"Currently using:\n{current}\n\n"
                f"Clear this and go back to the model bundled with the "
                f"installed package?",
            ) == QMessageBox.Yes:

                settings.remove(forcefield.CHGNET_PATH_KEY)
                forcefield.forget_calculators()

                self.log("CHGNet model reset to the bundled weights")

            return

        settings.setValue(forcefield.CHGNET_PATH_KEY, chosen)

        # Drop any calculator already built around the old weights,
        # otherwise the next analysis quietly runs the previous model.
        forcefield.forget_calculators()

        try:
            forcefield.get_calculator("chgnet")

        except forcefield.ForceFieldUnavailable as exc:
            QMessageBox.warning(self, "CHGNet model", str(exc))
            return

        self.log(f"CHGNet model set to {chosen}")

        QMessageBox.information(
            self,
            "CHGNet model",
            f"Loaded:\n{chosen}\n\nEvery force-field workflow will use "
            f"it, and exported figures will name it. This is remembered "
            f"between runs.",
        )

    # ------------------------------------------------------------
    # Electronegativity
    # ------------------------------------------------------------

    def electronegativity_polyhedral(self) -> None:
        """Open the readout with the coordination-shell view switched on."""

        self.electronegativity_panel(polyhedral=True)

    def electronegativity_panel(self, polyhedral: bool = False) -> None:
        """
        Open the live electronegativity readout for picked atoms.

        ``polyhedral`` picks which of the two views the panel opens on:
        the pair comparison, or the sum over the whole coordination
        shell. Select Atoms and Polyhedral Environment are two menu
        entries onto this one panel, and the entry is the only thing
        that says which view is wanted -- so it has to reach the panel,
        whether the panel is being built or is already on screen.
        """

        if self.workspace.viewer.structure is None:
            QMessageBox.information(
                self,
                "Electronegativity",
                "Open a structure file first.",
            )
            return

        existing = getattr(self, "_electronegativity_dialog", None)

        # Opening it twice would arm picking twice and leave two panels
        # showing the same selection.
        if existing is not None and existing.isVisible():
            # Switch the view first: raising a window that is already up
            # is invisible, so without this the menu entry looks like it
            # did nothing at all.
            existing.set_polyhedral(polyhedral)
            existing.raise_()
            existing.activateWindow()
            return

        # Held on the window: a non-modal dialog with no reference is
        # collected as soon as this method returns.
        self._electronegativity_dialog = ElectronegativityDialog(
            self.workspace.viewer, parent=self, polyhedral=polyhedral
        )

        self._electronegativity_dialog.show()

    def electronegativity_map(self) -> None:
        """Open the whole-structure map of electronegativity shift."""

        if self.workspace.viewer.structure is None:
            QMessageBox.information(
                self,
                "Electronegativity",
                "Open a structure file first.",
            )
            return

        from qvista.gui.dialogs.shift_map_window import ShiftMapWindow

        existing = getattr(self, "_shift_map_window", None)

        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            existing.rebuild()
            return

        # Held on the window, like the panel: a non-modal window with no
        # reference is collected as soon as this returns.
        self._shift_map_window = ShiftMapWindow(
            self.workspace.viewer, parent=self
        )

        self._shift_map_window.show()

    def electronegativity_clear(self) -> None:
        """Drop the picked atoms without closing the panel."""

        self.workspace.viewer.clear_selection()

    # ------------------------------------------------------------
    # Edit Structure
    # ------------------------------------------------------------

    def convert_structure(self) -> None:
        """
        Convert one structure file into another format.

        A file-in, file-out tool: it works on whatever path is chosen,
        so no project has to be open. The open structure is filled in
        as the source when there is one, since converting what is on
        screen is what this is nearly always opened for.
        """

        from qvista.gui.dialogs.convert_dialog import ConvertDialog

        dialog = ConvertDialog(start=self._run_file(), parent=self)

        dialog.exec()

        if dialog.result is not None:
            self.log(
                f"Converted to {dialog.result.target} "
                f"({dialog.result.formula}, "
                f"{dialog.result.natoms} atoms)"
            )

    def selective_dynamics(self) -> None:
        """
        Choose which atoms VASP is allowed to move.

        Still file-in, file-out -- freezing the bottom of a slab is done
        to a POSCAR on disk -- but the open structure is loaded into it
        to start with, and the viewer is handed over so the atoms can be
        picked by clicking rather than counted out of the file.

        Held on the window rather than run with ``exec``: picking needs
        the viewer behind it to stay clickable, and a modeless dialog
        with no reference is collected the moment this returns, taking
        its picking callback with it.
        """

        from qvista.gui.dialogs.selective_dialog import (
            SelectiveDynamicsDialog,
        )

        existing = getattr(self, "_selective_dialog", None)

        # Opening it twice would arm picking twice and leave two panels
        # fighting over the same selection.
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        self._selective_dialog = SelectiveDynamicsDialog(
            start=self._run_file(),
            viewer=self.workspace.viewer,
            parent=self,
        )

        self._selective_dialog.show()

    def remove_atom(self) -> None:
        """
        Delete atoms by clicking them in the viewer.

        Held on the window like the geometry panel is: a modeless
        dialog with no reference is collected the moment this returns,
        taking its picking callback with it.
        """

        from qvista.gui.dialogs.remove_atom_dialog import RemoveAtomDialog

        if self.workspace.viewer.structure is None:
            QMessageBox.information(
                self,
                "Remove Atom",
                "Open a structure file first.",
            )
            return

        existing = getattr(self, "_remove_atom_dialog", None)

        # Opening it twice would arm picking twice and leave two panels
        # fighting over the same selection.
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        self._remove_atom_dialog = RemoveAtomDialog(
            self.workspace.viewer, parent=self
        )

        self._remove_atom_dialog.show()

    def change_bond_length(self) -> None:
        """Edit the distance between two clicked atoms."""

        from qvista.gui.dialogs.geometry_dialog import LENGTH

        self._geometry_panel(LENGTH)

    def change_bond_angle(self) -> None:
        """Edit the angle between two bonds, at the middle atom."""

        from qvista.gui.dialogs.geometry_dialog import ANGLE

        self._geometry_panel(ANGLE)

    def _geometry_panel(self, mode: str) -> None:
        """Open the geometry editor, or bring it to the front."""

        from qvista.gui.dialogs.geometry_dialog import GeometryDialog

        if self.workspace.viewer.structure is None:
            QMessageBox.information(
                self,
                "Edit Structure",
                "Open a structure file first.",
            )
            return

        existing = getattr(self, "_geometry_dialog", None)

        # Opening it twice would arm picking twice and leave two panels
        # fighting over the same selection.
        if existing is not None and existing.isVisible():
            existing._select(mode)
            existing.raise_()
            existing.activateWindow()
            return

        # Held on the window: a non-modal dialog with no reference is
        # collected as soon as this method returns.
        self._geometry_dialog = GeometryDialog(
            self.workspace.viewer, mode=mode, parent=self
        )

        self._geometry_dialog.show()

    # ------------------------------------------------------------
    # VASP Input : POTCAR
    # ------------------------------------------------------------

    def potcar_pbe_hse(self) -> None:
        """Assemble a POTCAR from the standard PBE/HSE potentials."""

        self._potcar_dialog(PBE_FAMILY)

    def potcar_gw(self) -> None:
        """Assemble a POTCAR from the GW-specific potentials."""

        self._potcar_dialog(GW_FAMILY)

    # ------------------------------------------------------------
    # VASP Input : INCAR
    # ------------------------------------------------------------

    def incar_generator(self) -> None:
        """
        Assemble an INCAR for one of the eight kinds of calculation.

        A structure is not strictly needed -- an INCAR says nothing about
        the atoms -- but with one open the species order fixes the +U and
        LANGEVIN_GAMMA lists and the ion counts fix the band count, so it
        is asked for and then not insisted on.
        """

        from qvista.gui.dialogs.incar_dialog import IncarDialog

        elements: list[str] = []
        counts: list[int] = []

        project = self.project_controller.current_project()

        if project is not None:

            structure = self._prepare_vasp_structure(
                "INCAR", require_lattice=False
            )

            if structure is not None:
                elements = ordered_elements(structure)
                counts = species_counts(structure)

        IncarDialog(elements, counts, parent=self).exec()

    # ------------------------------------------------------------
    # VASP Input : K-points
    # ------------------------------------------------------------

    def kpoints_gamma_kspacing(self) -> None:
        """Gamma-centred mesh, sized from a KSPACING preset."""

        structure = self._prepare_vasp_structure("K-points")

        if structure is not None:
            self._open(KSpacingDialog(structure, GAMMA_SCHEME, parent=self))

    def kpoints_gamma_customised(self) -> None:
        """Gamma-centred mesh, with nx/ny/nz set by hand."""

        structure = self._prepare_vasp_structure("K-points")

        if structure is not None:
            self._open(CustomisedKpointsDialog(
                structure, GAMMA_SCHEME, parent=self
            ))

    def kpoints_mp_kspacing(self) -> None:
        """Monkhorst-Pack mesh, sized from a KSPACING preset."""

        structure = self._prepare_vasp_structure("K-points")

        if structure is not None:
            self._open(KSpacingDialog(structure, MP_SCHEME, parent=self))

    def kpoints_mp_customised(self) -> None:
        """Monkhorst-Pack mesh, with nx/ny/nz set by hand."""

        structure = self._prepare_vasp_structure("K-points")

        if structure is not None:
            self._open(CustomisedKpointsDialog(
                structure, MP_SCHEME, parent=self
            ))

    # ------------------------------------------------------------
    # Crystallography : High Symmetry Pathway
    # ------------------------------------------------------------

    def high_symmetry_pathway(self) -> None:
        """
        Determine the high-symmetry k-path.

        One dialog covers both the full recommended path and a custom
        build from this lattice's points. It stays open after a build --
        each request runs on the worker thread as usual, and the result is
        fed back into the still-open dialog, which draws the Brillouin
        zone, previews the KPOINTS, and offers to save them. Several
        builds can be tried before the dialog is closed by hand.
        """

        analysis = self._prepare_kpath()

        if analysis is None:
            return

        # The dialog needs the available points and the recommended path,
        # so seekpath runs here. The analysis caches it, so this is not
        # repeated on the worker thread.
        try:
            points = analysis.available_points()
            recommended = analysis.recommended_segments()

        except AnalysisError as exc:
            QMessageBox.critical(self, "High Symmetry Pathway", str(exc))
            return

        dialog = KPathDialog(points, recommended, parent=self)

        dialog.path_requested.connect(
            partial(self._build_kpath, analysis=analysis, dialog=dialog)
        )

        # Modeless, not exec(). The Brillouin zone is drawn into the
        # Plots window, which is a separate top-level window -- and an
        # application-modal dialog sits above every one of them, so the
        # figure appeared to vanish behind the dialog that had just
        # produced it. Kept on self so it is not garbage collected the
        # moment this method returns.
        self._kpath_dialog = dialog

        dialog.show()
        dialog.raise_()

    def _build_kpath(
        self,
        analysis: KPathAnalysis,
        dialog: KPathDialog,
    ) -> None:
        """Run the k-path build the dialog just requested."""

        analysis.npoints = dialog.npoints
        analysis.custom_segments = (
            None if dialog.use_recommended else dialog.custom_segments
        )

        self._run_analysis(
            analysis,
            on_success=partial(self._on_kpath_built, dialog=dialog),
        )

    def _on_kpath_built(self, result: object, dialog: KPathDialog) -> None:
        """Store the result as usual, then draw it into the open dialog."""

        self._store_kpath(result)

        if isinstance(result, KPathResult):
            dialog.show_result(result)

    def _open(self, dialog):
        """
        Show a dialog that draws into a figure window of its own.

        Modeless, deliberately. These dialogs put their plot in a
        separate top-level window, and an application-modal dialog sits
        above every other window in the application -- so the figure
        opened behind the dialog that produced it and the analysis
        looked like it had returned nothing. The table filled, the plot
        drew, and neither could be seen.

        Kept in a list because a dialog with no reference is collected
        the moment this returns, taking its figure window with it. They
        are dropped when closed, so opening the same analysis repeatedly
        does not accumulate them.
        """

        held = getattr(self, "_open_dialogs", None)

        if held is None:
            held = self._open_dialogs = []

        held.append(dialog)

        dialog.finished.connect(
            lambda _=0, d=dialog: d in held and held.remove(d)
        )

        dialog.show()
        dialog.raise_()

        return dialog

    def _prepare_kpath(self) -> KPathAnalysis | None:
        """
        Check preconditions and settle on a symmetry tolerance.

        Returns the configured analysis, or ``None`` if the structure is
        unsuitable or the user cancelled.
        """

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return None

        # Refuse a molecule before opening any dialog.
        try:
            require_periodic(project)

        except AnalysisError as exc:
            QMessageBox.warning(self, "High Symmetry Pathway", str(exc))
            return None

        analysis = KPathAnalysis(project)

        try:
            structure = analysis.structure()

        except AnalysisError as exc:
            QMessageBox.critical(self, "High Symmetry Pathway", str(exc))
            return None

        dialog = SymmetryPreviewDialog(
            lambda symprec: symmetry_preview(structure, symprec),
            analysis.symprec,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.statusBar().showMessage("Cancelled", 4000)
            return None

        analysis.symprec = dialog.symprec

        return analysis

    def _store_kpath(self, result: object) -> None:
        """Keep the path, and draw its Brillouin zone."""

        if not isinstance(result, KPathResult):
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.kpath = result.as_storage()

        self._plot_brillouin_zone(result)

    def _plot_brillouin_zone(self, result: KPathResult) -> None:
        """Draw the Brillouin zone and the chosen path into the plot dock."""

        from qvista.visualization import brillouin_plots

        try:
            brillouin_plots.draw_brillouin_zone(
                self.plot_window.take(),
                result,
                title=f"Brillouin zone - {result.spacegroup_symbol}",
            )

        except Exception as exc:
            self.log(f"Could not draw the Brillouin zone: {exc}")
            return

        self.plot_window.refresh()
        self.plot_window.present(self)

        self.log("")
        self.log(
            "Brillouin zone drawn in the Plots window. "
            "Drag to rotate; export the path with File > Export Data."
        )

    # ------------------------------------------------------------
    # Other : NEB and Molecular Dynamics
    # ------------------------------------------------------------

    def neb_analysis(self) -> None:
        """
        A machine-learned minimum energy path, start to finish.

        One window rather than the VASP route's two. That split exists
        because the calculation happens on another machine; a force
        field has no such gap, so splitting it would ask the user to
        walk through a door back into the same room.
        """

        from qvista.gui.dialogs.neb_chgnet_dialog import NebChgnetDialog

        start = ""

        project = self.project_controller.current_project()

        if project is not None and project.info.source_file is not None:
            start = str(project.info.source_file)

        NebChgnetDialog(start, parent=self).show()

    def relax_structure(self) -> None:
        """
        Relax the open structure with the machine-learning force field.

        The step before everything under Analysis: a band structure or a
        phonon spectrum computed on an unrelaxed cell describes a
        structure nobody has.
        """

        from qvista.analyses.relax import RelaxAnalysis
        from qvista.gui.dialogs.relax_dialog import RelaxDialog

        project = self._project_for_dynamics()

        if project is None:
            return

        structure = project.structure

        # Read the flag off structure_info, which is the one place that
        # knows: QVista's Structure wraps an ASE Atoms and has no
        # ``lattice`` of its own, so asking it for one answers "molecule"
        # for every crystal -- and greys out ISIF 3 and 4 for all of them.
        # This is the same source RelaxAnalysis checks, so the dialog can
        # no longer offer a mode the analysis will refuse, or refuse one
        # it would have accepted.
        dialog = RelaxDialog(
            natoms=len(structure.atoms) if structure else 0,
            periodic=bool(
                getattr(project.structure_info, "periodic", False)
            ),
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.statusBar().showMessage("Relaxation cancelled", 4000)
            return

        self._run_analysis(
            RelaxAnalysis(
                project,
                mode=dialog.mode,
                fmax=dialog.fmax,
                max_steps=dialog.steps,
            ),
            on_success=self._on_relax_done,
        )

    def _on_relax_done(self, result: object) -> None:
        """Store the relaxed structure, draw the trace, offer to keep it."""

        from qvista.analyses.relax import RelaxResult

        if not isinstance(result, RelaxResult):
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.relaxation = result.as_storage()

        from qvista.visualization import dynamics_plots

        try:
            dynamics_plots.draw_relaxation(
                self.plot_window.take(),
                result,
                title=(
                    f"{result.steps} steps, "
                    f"{result.energy_per_atom * 1000:.1f} meV/atom "
                    f"recovered"
                ),
            )

            self.plot_window.refresh()
            self.plot_window.present(self)

        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            self.log(f"Could not draw the relaxation trace: {exc}")

        self._offer_relaxed(result)

    def _offer_relaxed(self, result) -> None:
        """
        Ask whether to write the relaxed geometry out.

        Deliberately not written automatically. The input file is
        somebody's own, often the output of a calculation they care
        about, and a relaxation on a surrogate potential is not
        something to overwrite it with unasked.
        """

        if result.structure is None:
            return

        answer = QMessageBox.question(
            self,
            "Relaxation finished",
            f"Relaxed to {result.final_forces:.4f} eV/A in "
            f"{result.steps} steps.\n\n"
            f"Save the relaxed structure as a POSCAR?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if answer != QMessageBox.Yes:
            return

        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save relaxed structure", "CONTCAR_relaxed", ""
        )

        if not chosen:
            return

        try:
            Path(chosen).write_text(result.to_poscar(), encoding="utf-8")

        except OSError as exc:
            QMessageBox.critical(
                self, "Could not save", f"Writing failed: {exc}"
            )
            return

        self.log(f"Relaxed structure written to {chosen}")

    def neb_preprocess(self) -> None:
        """
        Write the image folders a VASP NEB needs.

        Takes two relaxed endpoints and interpolates between them; the
        run itself happens elsewhere.
        """

        from qvista.gui.dialogs.neb_vasp_dialog import NebPreprocessDialog

        dialog = NebPreprocessDialog(start=self._run_start(), parent=self)

        dialog.exec()

        if dialog.result is not None:
            self.log(
                f"NEB written: {dialog.result.total_folders} folders "
                f"under {dialog.result.root}"
            )

    def neb_postprocess(self) -> None:
        """Read the barrier out of a finished VASP NEB."""

        from qvista.gui.dialogs.neb_vasp_dialog import NebPostprocessDialog

        dialog = NebPostprocessDialog(start=self._run_start(), parent=self)

        # Modeless, so the profile and the structure viewer are both
        # reachable while it is open. The result is stored when it
        # closes rather than after exec() returns.
        def store(_code: int = 0) -> None:

            if dialog.result is None:
                return

            project = self.project_controller.current_project()

            if project is not None:
                project.results.neb = dialog.result.as_storage()

            self.log(
                f"NEB barrier {dialog.result.barrier_forward:.4f} eV"
            )

        dialog.finished.connect(store)

        self._open(dialog)

    def md_analysis(self) -> None:
        """Run molecular dynamics on the loaded structure."""

        project = self._project_for_dynamics()

        if project is None:
            return

        structure = project.structure

        dialog = MDDialog(
            natoms=len(structure.atoms) if structure else 0,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.statusBar().showMessage("Dynamics cancelled", 4000)
            return

        # Read off the dialog now: it is about to be discarded, and
        # _on_md_done runs long after it is gone.
        self._md_save = (
            (
                dialog.output_directory or self._default_md_output(project),
                dialog.save_choices,
            )
            if dialog.save_requested
            else None
        )

        self._run_analysis(
            MDAnalysis(
                project,
                temperature=dialog.temperature,
                timestep=dialog.timestep,
                steps=dialog.steps,
                ensemble=dialog.ensemble,
                friction=dialog.friction,
            ),
            on_success=self._on_md_done,
        )

    def _default_md_output(self, project) -> str:
        """
        Beside the structure that was run.

        Somewhere predictable and next to the input, rather than
        wherever a file dialog happened to be pointing last.
        """

        source = getattr(project.info, "source_file", None)

        if source is None:
            return ""

        return str(Path(source).parent / "mdexp")

    def _project_for_dynamics(self):
        """
        Shared preconditions for NEB and MD.

        Neither needs a lattice -- both work for molecules -- but both need
        a structure and a force field.
        """

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return None

        if project.info.source_file is None:
            QMessageBox.warning(
                self,
                "No structure",
                "This project has no structure file to run.",
            )
            return None

        if not forcefield.is_available():
            QMessageBox.warning(
                self,
                "Force field not available",
                forcefield.unavailable_message(),
            )
            return None

        return project

    # ------------------------------------------------------------

    def _on_neb_done(self, result: object) -> None:
        """Store the path and draw its energy profile."""

        if not isinstance(result, NEBResult):
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.neb = result.as_storage()

        from qvista.visualization import dynamics_plots

        try:
            dynamics_plots.draw_neb_profile(
                self.plot_window.take(),
                result,
                title="Minimum energy path",
            )

        except Exception as exc:
            self.log(f"Could not draw the path: {exc}")
            return

        self.plot_window.refresh()
        self.plot_window.present(self)

    def _session_trajectory(self):
        """
        The trajectory this session's CHGNet run recorded, if any.

        Returns ``None`` when no run has been done, when the stored run
        came from VASP rather than from here, or when it is too short to
        be a trajectory. Everything under MD > CHGNet > Analysis opens
        on this; everything under MD > VASP > Analysis reads a file
        instead, which is why the two branches do not share a slot.
        """

        from qvista.services import msd as msd_service

        project = self.project_controller.current_project()

        stored = (
            None if project is None else project.results.md.get("result")
        )

        if not isinstance(stored, MDResult) or len(stored.frames) <= 1:
            return None

        # Reading a VASP run stores an MDResult in the same place, so
        # without this the CHGNet analyses would open on it and label
        # the figure with the wrong method.
        from qvista.analyses.md import ORIGIN_FORCEFIELD

        if stored.origin != ORIGIN_FORCEFIELD:
            return None

        try:
            return msd_service.from_frames(
                stored.frames,
                stored.symbols,
                lattice=stored.cell,
                # The run records every so many steps, so the frames are
                # further apart than the integration timestep.
                timestep=stored.timestep
                * max(1, stored.steps // max(1, len(stored.frames)))
                / 1000.0,
                temperature=stored.temperature,
            )

        except msd_service.TrajectoryError as exc:
            self.log(f"Could not use the recorded trajectory: {exc}")

        return None

    def _no_session_run(self, what: str) -> bool:
        """
        Say so when there is nothing from CHGNet to analyse.

        Without this the window opens empty and looks broken, when what
        actually happened is that the dynamics have not been run.
        """

        QMessageBox.information(
            self,
            what,
            "There is no CHGNet trajectory in this session yet.\n\n"
            "Run MD > CHGNet > Run Dynamics first, or read a finished "
            "VASP run through MD > VASP > Analysis.",
        )

        return True

    def md_diffusion(self) -> None:
        """
        Mean squared displacement and diffusion from an XDATCAR.

        The VASP branch: a finished AIMD run read off disk, which is the
        commoner source and never passed through QVista.
        """

        from qvista.gui.dialogs.msd_dialog import MsdDialog

        self._open(MsdDialog(
            trajectory=None, start=self._trajectory_start(), parent=self
        ))

    def chgnet_msd(self) -> None:
        """The same analysis on the run this session just did."""

        from qvista.gui.dialogs.msd_dialog import MsdDialog

        trajectory = self._session_trajectory()

        if trajectory is None and self._no_session_run(
            "MSD and Diffusion"
        ):
            return

        self._open(MsdDialog(
            trajectory=trajectory,
            start=self._trajectory_start(),
            parent=self,
        ))

    def md_analysis_vasp(self) -> None:
        """
        Read a finished VASP ab-initio molecular-dynamics run.

        The traces come from OSZICAR and the trajectory from XDATCAR, so
        this works on a run killed by a wall clock as well as one that
        finished -- which is most of them.
        """

        from qvista.gui.dialogs.md_vasp_dialog import MdVaspDialog

        dialog = MdVaspDialog(start=self._run_start(), parent=self)

        def store(_code: int = 0) -> None:

            if dialog.result is None:
                return

            project = self.project_controller.current_project()

            if project is not None:
                project.results.md = dialog.result.as_storage()

            self.log(f"VASP MD read from {dialog.run.directory}")

        dialog.finished.connect(store)

        self._open(dialog)

    # ------------------------------------------------------------
    # MD : correlations from a finished trajectory
    # ------------------------------------------------------------

    def md_rdf(self) -> None:
        """Radial distribution function from an XDATCAR."""

        self._correlations("rdf")

    def md_van_hove(self) -> None:
        """Van Hove correlation, self and distinct parts."""

        self._correlations("vanhove")

    def md_residence(self) -> None:
        """How long an ion stays put before it hops."""

        self._correlations("residence")

    def _correlations(self, mode: str, trajectory=None) -> None:
        """
        Open the correlations window on one analysis.

        All three read the same trajectory and differ only in what they
        ask of it, so they share a window and it is parsed once rather
        than per analysis.
        """

        from qvista.gui.dialogs.correlations_dialog import (
            CorrelationsDialog,
        )

        self._open(CorrelationsDialog(
            start=self._trajectory_start(),
            mode=mode,
            trajectory=trajectory,
            parent=self,
        ))

    # ----------------------------------------------------------
    # The same three on this session's CHGNet run
    # ----------------------------------------------------------

    def chgnet_rdf(self) -> None:
        """Radial distribution function from the CHGNet run."""

        self._chgnet_correlations("rdf", "RDF")

    def chgnet_van_hove(self) -> None:
        """Van Hove correlation from the CHGNet run."""

        self._chgnet_correlations("vanhove", "Van Hove Correlation")

    def chgnet_residence(self) -> None:
        """Residence time from the CHGNet run."""

        self._chgnet_correlations("residence", "Residence Time")

    def _chgnet_correlations(self, mode: str, what: str) -> None:

        trajectory = self._session_trajectory()

        if trajectory is None and self._no_session_run(what):
            return

        self._correlations(mode, trajectory=trajectory)

    def md_kinetics(self) -> None:
        """
        Activation energy, prefactor and ionic conductivity.

        Needs one trajectory per temperature: a single run gives a
        single diffusivity, and an activation energy is a slope.
        """

        from qvista.gui.dialogs.kinetics_dialog import KineticsDialog

        self._open(KineticsDialog(
            start=self._trajectory_start(), parent=self
        ))

    def chgnet_kinetics(self) -> None:
        """
        The same fit over CHGNet runs.

        One run is one temperature and so one point; the slope needs
        several, which means several runs saved to folders of their own.
        The window reads them the same way it reads VASP runs, so this
        opens the same dialog -- the difference is only where the
        trajectories came from.
        """

        self.md_kinetics()

    def _trajectory_start(self) -> str:
        """A sensible folder for a file browser looking for an XDATCAR."""

        return self._run_start()

    def _on_md_done(self, result: object) -> None:
        """Store the run and draw its traces."""

        if not isinstance(result, MDResult):
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.md = result.as_storage()

        from qvista.visualization import dynamics_plots

        try:
            dynamics_plots.draw_md_traces(
                self.plot_window.take(),
                result,
                title=(
                    f"{result.ensemble.upper()} at "
                    f"{result.temperature:.0f} K"
                ),
            )

        except Exception as exc:
            self.log(f"Could not draw the traces: {exc}")
            return

        self.plot_window.refresh()
        self.plot_window.present(self)

        self._save_md_run(result)

    def _save_md_run(self, result) -> None:
        """
        Write the finished run out, if that was asked for.

        After the traces are drawn rather than before, because the plot
        is one of the things being saved and does not exist until then.
        A failure here is logged and does not disturb the result already
        on screen: the dynamics happened either way, and losing the
        window over a full disk would be the worse outcome.
        """

        wanted = getattr(self, "_md_save", None)

        if not wanted:
            return

        self._md_save = None

        folder, choices = wanted

        if not folder:
            self.log(
                "The run was not saved: there was nowhere to put it."
            )
            return

        from qvista.services import md_vasp

        try:
            saved = md_vasp.save_run(
                result,
                folder,
                figure=self.plot_window.figure,
                **choices,
            )

        except Exception as exc:  # noqa: BLE001 - the run stands
            self.log(f"The run finished but could not be saved: {exc}")
            return

        self.log(f"Saved {saved.count} files to {saved.root}")

        for note in saved.skipped:
            self.log(f"  not written -- {note}")

        self.statusBar().showMessage(
            f"Dynamics saved to {saved.root}", 6000
        )

    # ------------------------------------------------------------
    # Other : Mechanical Properties
    # ------------------------------------------------------------

    def mechanical_properties(self) -> None:
        """Relax, strain, and fit the elastic tensor."""

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return

        try:
            require_periodic(project)

        except AnalysisError as exc:
            QMessageBox.warning(self, "Mechanical Properties", str(exc))
            return

        if not forcefield.is_available():
            QMessageBox.warning(
                self,
                "Force field not available",
                forcefield.unavailable_message(),
            )
            return

        # ----------------------------------------------------------
        # Symmetry matters here: an elastic tensor is only meaningful in a
        # standard orientation, and refinement cannot supply one for a P1
        # structure. Let the user see and retune the tolerance first.
        # ----------------------------------------------------------

        analysis = ElasticAnalysis(project)

        try:
            structure = analysis.structure()

        except AnalysisError as exc:
            QMessageBox.critical(self, "Mechanical Properties", str(exc))
            return

        dialog = SymmetryPreviewDialog(
            lambda symprec: symmetry_preview(structure, symprec),
            analysis.symprec,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.statusBar().showMessage("Cancelled", 4000)
            return

        analysis.symprec = dialog.symprec

        self._run_analysis(analysis, on_success=self._on_elastic_done)

    def mechanical_properties_vasp(self) -> None:
        """
        The elastic tensor from a finished VASP run.

        Nothing is computed here: VASP already did it with IBRION = 6,
        and this reads the tensor out of OUTCAR. The result goes to the
        same panel the CHGNet route uses, so the two can be compared
        side by side.
        """

        from qvista.gui.dialogs.elastic_vasp_dialog import ElasticVaspDialog

        dialog = ElasticVaspDialog(start=self._run_start(), parent=self)

        if dialog.exec() != QDialog.Accepted or dialog.result is None:
            return

        self._on_elastic_done(dialog.result)

        self.log(
            f"Elastic tensor read from {dialog.elastic.source}"
        )

    def _run_start(self) -> str:
        """The folder to open a file browser on, if anything is loaded."""

        project = self.project_controller.current_project()

        if project is not None and project.info.source_file:
            return str(Path(project.info.source_file).parent)

        return ""

    def _run_file(self) -> str:
        """
        The structure file currently open, if there is one.

        For the tools that work on a path rather than on the project:
        handing them the loaded file spares the user finding it again on
        disk, which is the same file they picked to open QVista on.
        Falls back to the folder, which is all the browser needs.
        """

        project = self.project_controller.current_project()

        if project is None:
            return self._run_start()

        source = project.info.source_file

        if source is not None and Path(source).is_file():
            return str(source)

        # A project opened as a folder has no source file. The POSCAR
        # beside it is what was drawn -- ProjectService picks the same
        # one -- so it is what these tools should start on.
        poscar = project.info.files.get("POSCAR")

        if poscar is not None and Path(poscar).is_file():
            return str(poscar)

        return self._run_start()

    def _on_elastic_done(self, result: object) -> None:
        """Store the tensor and show it in the panel."""

        if not isinstance(result, ElasticResult):
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.elastic = result.as_storage()

        self.elastic_window.show_result(result)
        self.elastic_window.present(self)

    # ------------------------------------------------------------
    # Phonon Analysis : CHGNet
    #
    # Forces come from the installed machine-learning potential, so
    # everything below runs inside QVista and answers in seconds. The
    # VASP branch further down is the same phonopy pipeline with the
    # forces coming from a first-principles calculation instead.
    # ------------------------------------------------------------

    def phonon_band(self) -> None:
        """Phonon band structure along the high-symmetry path."""

        self._run_phonon(MODE_BAND)

    def phonon_dos(self) -> None:
        """
        The density of states section, over cached force constants.

        The forces are the expensive part and the DOS is seconds of
        post-processing on top, so this opens a panel that re-runs
        rather than a single figure with the settings baked in. When
        there are no force constants yet it falls back to computing
        them the same way the band structure does; the panel opens on
        the result.
        """

        project = self.project_controller.current_project()

        if self._any_phonon(project) is not None:
            self._open_dos_panel()
            return

        if project is None:
            QMessageBox.information(
                self,
                "No force constants",
                "This works on force constants from either route.\n\n"
                "Open a structure and run Phonon Analysis > CHGNet > "
                "Phonon Band, or run Phonon Analysis > VASP > "
                "Post-Processing on a finished configuration.",
            )
            return

        self._run_phonon(MODE_DOS, then=self._open_dos_panel)

    def _any_phonon(self, project: Project | None):
        """
        Cached force constants, whichever route produced them.

        Deliberately looser than :meth:`_cached_phonon`, which insists
        the force constants belong to the structure on screen. Mode
        animation needs that, because it moves the atoms being
        displayed; a density of states does not touch the geometry at
        all, and refusing to show one because a different file happens
        to be open would make the VASP route unusable -- post-processing
        is meant to work on a folder computed elsewhere, with no
        matching structure loaded.
        """

        if project is None:
            return None

        cached = project.results.phonon.get("data")

        return cached if isinstance(cached, PhononData) else None

    def _open_dos_panel(self) -> None:
        """Show the DOS dialog on whatever force constants are cached."""

        project = self.project_controller.current_project()

        cached = self._any_phonon(project)

        if cached is None:
            return

        from qvista.gui.dialogs.phonon_dos_dialog import PhononDosDialog

        dialog = PhononDosDialog(
            cached.phonon,
            force_field=cached.force_field,
            parent=self,
        )

        self._open(dialog)

    def phonon_band_with_dos(self) -> None:
        """Band structure and DOS side by side, sharing the frequency axis."""

        self._run_phonon(MODE_BOTH)

    def phonon_thermal(self) -> None:
        """Free energy, entropy and heat capacity against temperature."""

        self._run_phonon(MODE_THERMAL)

    def projected_phonon_band(self) -> None:
        """Phonon bands with each mode projected onto the elements."""

        self._run_phonon(MODE_PROJECTED)

    def phonon_band_visualization(self) -> None:
        """Animate a normal mode in the 3D viewer."""

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return

        cached = self._cached_phonon(project)

        if cached is None:
            QMessageBox.information(
                self,
                "No force constants",
                "Mode animation needs force constants.\n\n"
                "Run Phonon Analysis > CHGNet > Phonon Band first; the result is "
                "cached and reused here.",
            )
            return

        self._animate_modes(cached.phonon)

    # ------------------------------------------------------------

    def _animate_modes(self, phonon) -> None:
        """
        Open the mode animation on a phonopy object, from either route.

        The primitive cell is taken from the force constants being
        animated rather than looked up from the project, so this works
        on a configuration computed on another machine with nothing
        matching open here -- which is the whole point of the VASP
        branch.
        """

        project = self.project_controller.current_project()

        kpath = (
            None if project is None else project.results.kpath.get("result")
        )

        points = (
            kpath.points
            if isinstance(kpath, KPathResult) and kpath.points
            else {"GAMMA": [0.0, 0.0, 0.0]}
        )

        primitive = phonon.primitive

        # No periodic images while a mode plays. This dialog animates a
        # single primitive cell with no exp(i q.R) phase in it, so every
        # image the boundary tiles carries the home cell's displacement
        # exactly. At any q away from Gamma neighbouring cells are meant
        # to be out of phase, so the copies move wrongly -- and at the
        # zone boundary, where they should be in antiphase, they move
        # precisely backwards. Restored in :meth:`_restore_structure`.
        self.workspace.viewer.tile_images = False

        dialog = ModeAnimationDialog(
            phonon,
            points,
            on_frame=lambda positions: self._show_mode_frame(
                positions, primitive
            ),
            on_stop=self._restore_structure,
            parent=self,
        )

        dialog.export_button.clicked.connect(
            lambda: self._export_mode_trajectory(dialog)
        )

        self.log("")
        self.log(
            "Mode animation: pick a wavevector and band, then press Play. "
            "The viewer shows the primitive cell while animating."
        )

        dialog.show()

    def _show_mode_frame(self, positions, primitive) -> None:
        """Display one animation frame in the 3D viewer."""

        from ase import Atoms

        from qvista.models.structure import Structure

        if primitive is None:
            return

        try:
            atoms = Atoms(
                symbols=[str(s) for s in primitive.symbols],
                positions=positions,
                cell=primitive.cell,
                pbc=True,
            )

            self.workspace.load_structure(Structure(atoms))

        except Exception as exc:
            self.log(f"Animation stopped: {exc}")

    def _restore_structure(self) -> None:
        """Put the original structure back after animating."""

        # Whatever else happens, the viewer goes back to tiling: an
        # animation that ends must not leave the boundary controls
        # silently doing nothing.
        self.workspace.viewer.tile_images = True

        project = self.project_controller.current_project()

        if project is None or project.structure is None:
            return

        try:
            self.workspace.load_structure(project.structure)

        except Exception:
            pass

    def _export_mode_trajectory(self, dialog) -> None:
        """Write the animated mode as an extended XYZ trajectory."""

        from qvista.analyses.phonon_modes import to_xyz_trajectory

        mode = dialog.current_mode

        if mode is None or not dialog.frames:
            return

        primitive = dialog.phonon.primitive

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export mode trajectory",
            f"mode_{mode.label}_band{mode.band_index + 1}.xyz",
            "Extended XYZ (*.xyz);;All Files (*)",
        )

        if not filename:
            return

        text = to_xyz_trajectory(
            [str(s) for s in primitive.symbols],
            dialog.frames,
            cell=primitive.cell,
            comment=(
                f"QVista phonon mode {mode.label} band "
                f"{mode.band_index + 1} f={mode.frequency:.4f}THz"
            ),
        )

        try:
            Path(filename).write_text(text, encoding="utf-8")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export failed",
                f"Could not write {filename}: {exc}",
            )
            return

        self.log(f"Exported mode trajectory to {filename}")

    def phonon_mode_explorer(self) -> None:
        """
        Open the dispersion beside the cell it describes.

        Needs a band structure to click on, so it runs on whatever the
        project already holds -- from either the CHGNet route or the
        VASP one -- rather than recomputing.
        """

        from qvista.gui.dialogs.mode_explorer import ModeExplorer

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self, "Phonon modes", "Open a structure file first."
            )
            return

        data = project.results.phonon.get("data")
        band = project.results.phonon.get("band")

        if not isinstance(data, PhononData) or not band:
            QMessageBox.information(
                self,
                "Phonon modes",
                "Compute a phonon band structure first.\n\n"
                "Phonon Analysis > CHGNet > Phonon Band for a "
                "quick answer, or the VASP route for a first-principles "
                "one. Either fills this in.",
            )
            return

        existing = getattr(self, "_mode_explorer", None)

        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        # Held on the window: a non-modal window with no reference is
        # collected as soon as this method returns.
        self._mode_explorer = ModeExplorer(
            data.phonon,
            band,
            force_field=data.force_field,
            parent=self,
        )

        self._mode_explorer.show()

    def _run_phonon(self, mode: str, then=None) -> None:
        """
        Shared entry point for the phonon menu items.

        The expensive part -- displacements and forces -- runs once. When
        matching force constants are already cached on the project, the
        other menu items reuse them and only re-run the cheap
        post-processing, so no dialog is shown at all.

        ``then`` runs after the result has been stored, which is how the
        DOS panel gets opened on force constants that did not exist when
        the menu item was clicked.
        """

        def done(result: object) -> None:

            self._on_phonon_done(result)

            if then is not None:
                then()

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self,
                "No project",
                "Open a structure file first.",
            )
            return

        try:
            require_periodic(project)

        except AnalysisError as exc:
            QMessageBox.warning(self, "Phonon calculation", str(exc))
            return

        kpath = self._get_kpath(project)

        if kpath is None:
            return

        # ----------------------------------------------------------
        # Reuse cached force constants when they match
        # ----------------------------------------------------------

        cached = self._cached_phonon(project)

        if cached is not None:

            self._run_analysis(
                PhononAnalysis(
                    project,
                    kpath,
                    supercell=cached.supercell,
                    symprec=cached.symprec,
                    distance=cached.distance,
                    mode=mode,
                    cached=cached,
                ),
                on_success=done,
            )
            return

        # ----------------------------------------------------------
        # A fresh run needs the force field
        # ----------------------------------------------------------

        if not forcefield.is_available():
            QMessageBox.warning(
                self,
                "Force field not available",
                forcefield.unavailable_message(),
            )
            return

        analysis = self._configure_phonon(project, kpath, mode)

        if analysis is None:
            return

        self._run_analysis(analysis, on_success=done)

    # ------------------------------------------------------------

    def _get_kpath(self, project: Project) -> KPathResult | None:
        """
        Return the project's k-path, computing it if necessary.

        Reuses the stored path so the phonon workflow and the
        Crystallography menu agree on the reference cell.
        """

        stored = project.results.kpath.get("result")

        if isinstance(stored, KPathResult):

            # Said every time, because the path decides what the figure
            # shows and nothing on the figure says where it came from.
            self.log(
                f"Band path: {_path_label(stored.segments)}"
                f"   ({'custom' if stored.is_custom else 'recommended'})"
                "   -- change it on the plot, with Select Custom Path."
            )

            return stored

        try:
            result = KPathAnalysis(project).run()

        except AnalysisError as exc:
            QMessageBox.critical(self, "High Symmetry Pathway", str(exc))
            return None

        project.results.kpath = result.as_storage()

        return result

    def _cached_phonon(self, project: Project):
        """Return cached force constants for this structure, if any."""

        cached = project.results.phonon.get("data")

        if not isinstance(cached, PhononData):
            return None

        source = project.info.source_file

        if cached.source != ("" if source is None else str(source)):
            return None

        return cached

    def _configure_phonon(
        self,
        project: Project,
        kpath: KPathResult,
        mode: str,
    ):
        """
        Show the guards and the supercell dialog for a fresh run.

        Returns the configured analysis, or ``None`` if the structure is
        unsuitable or the user cancelled.
        """

        primitive = kpath.primitive

        if primitive is None:
            QMessageBox.critical(
                self,
                "Phonon calculation",
                "The k-path carried no primitive cell to build on.",
            )
            return None

        structure = primitive.to_pymatgen()

        # ----------------------------------------------------------
        # Guards
        # ----------------------------------------------------------

        warnings: list[str] = []

        try:
            preview = symmetry_preview(structure, kpath.symprec)
            warnings.extend(preview.warnings)

        except AnalysisError as exc:
            QMessageBox.critical(self, "Phonon calculation", str(exc))
            return None

        vacuum = detect_vacuum(structure)
        warnings.extend(vacuum.warnings)

        # ----------------------------------------------------------
        # Supercell
        # ----------------------------------------------------------

        lattice = structure.lattice

        dialog = SupercellDialog(
            formula=structure.composition.reduced_formula,
            primitive_natoms=len(structure),
            lattice_abc=lattice.abc,
            lattice_angles=lattice.angles,
            primitive_volume=lattice.volume,
            count_displacements=lambda supercell: count_displacements(
                primitive,
                supercell,
                kpath.symprec,
            ),
            suggested=suggest_supercell(len(structure), vacuum),
            warnings=warnings,
            parent=self,
        )

        if dialog.exec() != QDialog.Accepted:
            self.statusBar().showMessage("Phonon run cancelled", 4000)
            return None

        return PhononAnalysis(
            project,
            kpath,
            supercell=dialog.supercell,
            symprec=kpath.symprec,
            mode=mode,
        )

    # ------------------------------------------------------------

    def _on_phonon_done(self, result: object) -> None:
        """Cache the force constants and draw the requested figure."""

        if not isinstance(result, PhononResult):
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.phonon = result.as_storage()

        self._plot_phonon(result)

    def _plot_phonon(self, result: PhononResult) -> None:
        """
        Draw the phonon figure into the plot window.

        The result is kept so the branch colouring can be changed
        without recomputing anything: the force constants took minutes
        and the figure takes milliseconds, so a colour scheme must not
        cost a re-run.
        """

        self._phonon_plotted = result

        self._draw_phonon()

        self.plot_window.present(self)

    def _draw_phonon(self) -> None:
        """Redraw the last phonon result with the current band controls."""

        from qvista.visualization import phonon_plots

        result = getattr(self, "_phonon_plotted", None)

        if result is None:
            return

        force_field = (
            "" if result.phonon_data is None else result.phonon_data.force_field
        )

        # Every figure that draws a dispersion takes the controls, so
        # that the path can be changed from any of them. A DOS on its
        # own has no dispersion, and neither has the thermal panel.
        wants_controls = bool(result.band) and result.mode != MODE_THERMAL

        # The projected figure colours by element and legends itself, so
        # its branch controls would do nothing -- but the path button
        # applies exactly as much as it does anywhere else.
        colouring = result.mode != MODE_PROJECTED

        bands = self._band_controls() if wants_controls else None

        if bands is not None:
            bands.set_colouring_offered(colouring)

        figure = self.plot_window.take(bands)

        style = bands.values if bands is not None and colouring else {}

        try:
            if result.mode == MODE_THERMAL:

                thermal = getattr(result, "thermal", None)

                if thermal is None:
                    self.log(
                        "No thermal properties in this result. They are "
                        "integrated over the DOS mesh, so the DOS has to "
                        "have run."
                    )
                    return

                phonon_plots.draw_thermal(
                    figure,
                    thermal,
                    force_field=force_field,
                    title="Harmonic thermal properties",
                )

            elif result.mode == MODE_PROJECTED and result.band:
                phonon_plots.draw_projected_band(
                    figure,
                    result.band,
                    force_field=force_field,
                    title="Element-projected phonon bands",
                )

            elif result.mode == MODE_BOTH and result.band and result.dos:
                phonon_plots.draw_band_with_dos(
                    figure,
                    result.band,
                    result.dos,
                    force_field=force_field,
                    **style,
                )

            elif result.band:
                phonon_plots.draw_band(
                    figure,
                    result.band,
                    force_field=force_field,
                    **style,
                )

            elif result.dos:
                phonon_plots.draw_dos(
                    figure,
                    result.dos,
                    force_field=force_field,
                )

            else:
                return

        except Exception as exc:
            self.log(f"Could not draw the figure: {exc}")
            return

        self.plot_window.refresh()

    def _band_controls(self):
        """
        The dispersion's colour controls, built once and reused.

        Kept on the window rather than on the plot window because the
        chosen scheme should survive re-running the analysis, in the
        same way a chosen Design does.
        """

        existing = getattr(self, "_phonon_band_controls", None)

        if existing is not None:
            return existing

        from qvista.gui.widgets.band_controls import BandControls

        controls = BandControls()
        controls.changed.connect(self._draw_phonon)
        controls.path_requested.connect(self._choose_phonon_path)

        self._phonon_band_controls = controls

        return controls

    # ------------------------------------------------------------
    # The path a dispersion is drawn along
    # ------------------------------------------------------------

    def _choose_phonon_path(self) -> None:
        """
        Pick the high-symmetry path, and redraw the dispersion along it.

        Opened from the figure, because that is where the question comes
        up. The force constants are already built by this point, so a
        different path costs one diagonalisation per q-point -- seconds
        against the hours they took -- and nothing is re-run.

        The points come from the primitive cell phonopy derived, not
        from whatever structure happens to be open. Post-processing is
        meant to work on a folder someone else produced, where there may
        be no open structure at all, and a path expressed in another
        cell's reciprocal basis would be wrong rather than rejected.
        """

        from qvista.gui.dialogs.phonon_path_dialog import PhononPathDialog
        from qvista.services import phonon_path

        result = getattr(self, "_phonon_plotted", None)

        phonon = None

        if result is not None and result.phonon_data is not None:
            phonon = getattr(result.phonon_data, "phonon", None)

        if phonon is None:
            QMessageBox.information(
                self,
                "No force constants",
                "Changing the path re-evaluates the dynamical matrix, "
                "so it needs the force constants the figure was drawn "
                "from.",
            )
            return

        try:
            options = phonon_path.options_for(phonon.primitive)

        except phonon_path.PhononPathError as exc:
            QMessageBox.warning(self, "Band path", str(exc))
            return

        dialog = PhononPathDialog(
            options,
            current=getattr(self, "_phonon_path_segments", None),
            parent=self,
        )

        dialog.path_chosen.connect(
            lambda segments, npoints: self._apply_phonon_path(
                options, segments, npoints
            )
        )

        dialog.exec()

    def _apply_phonon_path(self, options, segments, npoints: int) -> None:
        """Re-run the dispersion along ``segments`` and redraw."""

        from qvista.services import phonon_path

        result = getattr(self, "_phonon_plotted", None)

        if result is None or result.phonon_data is None:
            return

        phonon = getattr(result.phonon_data, "phonon", None)

        if phonon is None:
            return

        self.log(
            f"Re-running the dispersion along "
            f"{phonon_path.path_label(segments)} "
            f"({npoints} points per segment)..."
        )

        try:
            band = phonon_path.run_band(
                phonon, options.points, segments, npoints=npoints
            )

        except phonon_path.PhononPathError as exc:
            QMessageBox.warning(self, "Band path", str(exc))
            self.log(f"Band path failed: {exc}")
            return

        # Remembered so the chooser reopens on what is on screen rather
        # than on the recommended path every time.
        self._phonon_path_segments = list(segments)

        result.band = band

        # The element projection is rebuilt from the new eigenvectors,
        # or the projected figure would still be coloured by the old
        # path's modes.
        self._reproject_phonon(result)

        self._draw_phonon()

        self.log("Done.")

    def _reproject_phonon(self, result) -> None:
        """Recompute the element projection for a freshly run band."""

        from qvista.analyses.phonon import project_eigenvectors

        band = result.band

        if not isinstance(band, dict) or "eigenvectors" not in band:
            return

        phonon = getattr(result.phonon_data, "phonon", None)

        if phonon is None:
            return

        try:
            band["symbols"] = [
                str(symbol) for symbol in phonon.primitive.symbols
            ]

            band["projections"] = project_eigenvectors(
                band["eigenvectors"], band["symbols"]
            )

        except Exception as exc:  # noqa: BLE001 - the band still stands
            self.log(f"Projection skipped: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Phonon Analysis : VASP
    #
    # The same phonopy pipeline as the CHGNet branch above, cut in half:
    # QVista writes the VASP runs, the user runs them somewhere else,
    # and QVista reads them back. Everything that has to survive in
    # between is in the configuration folder, so the two halves do not
    # depend on the same session or even the same machine.
    # ------------------------------------------------------------

    def vasp_phonon_preprocess(self) -> None:
        """
        Dimensions, then a method, then the inputs.

        Three clicks and no prose: everything else -- the cutoff, the
        k-mesh, the pseudopotentials -- is derived, and anything that
        goes wrong arrives as a message box.
        """

        from qvista.gui.dialogs.phonon_vasp_dialog import (
            DimensionsDialog,
            PhononVaspDialog,
            confirm_size,
        )

        structure, symprec, suggested = self._phonon_reference()

        if structure is None:
            return

        # ----------------------------------------------------------
        # Dimensions
        # ----------------------------------------------------------

        dimensions = DimensionsDialog(suggested, parent=self)

        if dimensions.exec() != QDialog.Accepted:
            return

        supercell = dimensions.supercell

        nx, ny, nz = supercell

        if not confirm_size(len(structure) * nx * ny * nz, parent=self):
            return

        # ----------------------------------------------------------
        # Generate, which is the slow part
        # ----------------------------------------------------------

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            phonon = phonopy_vasp.build_phonon(
                structure, supercell, symprec
            )

            count = len(phonon.supercells_with_displacements)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Phonopy",
                f"Could not build the supercell.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        finally:
            QApplication.restoreOverrideCursor()

        if count == 0:
            QMessageBox.warning(
                self,
                "Phonopy",
                "phonopy generated no displacements for this cell.",
            )
            return

        # ----------------------------------------------------------
        # Method, then inputs
        # ----------------------------------------------------------

        dialog = PhononVaspDialog(
            phonon,
            formula=structure.composition.reduced_formula,
            parent=self,
        )

        dialog.exec()

        if dialog.config is None:
            return

        project = self.project_controller.current_project()

        if project is not None:
            project.results.phonon_vasp = {"configuration": dialog.config}

        self.log("")
        self.log(f"Phonopy - VASP ({dialog.config.method})")
        self.log("----------------")

        for label, value in dialog.config.rows():
            self.log(f"{label:<19}: {value}")

        for text in dialog.config.warnings:
            self.log(text)

        self.statusBar().showMessage(
            f"Wrote {len(dialog.config.force_runs)} run folders to "
            f"{dialog.config.root}",
            8000,
        )

    def _phonon_reference(self):
        """
        ``(structure, symprec, suggested supercell)`` for a phonon run.

        The structure is the one that was opened, exactly as it is.
        That is phonopy's own workflow -- the unit cell is whatever is
        in the POSCAR, and ``primitive_matrix="auto"`` leaves phonopy to
        reduce it and to take the band path from the cell it reduced to.
        Nothing is standardised on the way in.

        Returns ``(None, 0.0, ())`` once it has explained why not.
        """

        project = self.project_controller.current_project()

        if project is None:
            QMessageBox.information(
                self, "No project", "Open a structure file first."
            )
            return None, 0.0, ()

        try:
            require_periodic(project)

        except AnalysisError as exc:
            QMessageBox.warning(self, "Phonopy", str(exc))
            return None, 0.0, ()

        structure = self._prepare_vasp_structure("Phonopy")

        if structure is None:
            return None, 0.0, ()

        symprec = DEFAULT_SYMPREC

        # Guards are worth interrupting for -- a vacuum direction
        # repeated is empty space costing as much as material -- but not
        # worth a panel of text, so they arrive as one box.
        warnings: list[str] = []

        try:
            warnings.extend(
                symmetry_preview(structure, symprec).warnings
            )

        except AnalysisError as exc:
            QMessageBox.critical(self, "Phonopy", str(exc))
            return None, 0.0, ()

        vacuum = detect_vacuum(structure)
        warnings.extend(vacuum.warnings)

        if warnings:
            QMessageBox.warning(self, "Phonopy", "\n\n".join(warnings))

        return (
            structure,
            symprec,
            suggest_supercell(len(structure), vacuum),
        )

    # ------------------------------------------------------------
    # Phonon Analysis : VASP : Post-Processing
    #
    # The same four figures the CHGNet branch offers, over first
    # principles forces. Reading the configuration back is the expensive
    # half and it produces the band, the density of states and the
    # element projections in one pass, so the four entries are four
    # readings of one result rather than four runs: the first click asks
    # for the folder and the rest are redraws.
    # ------------------------------------------------------------

    def vasp_phonon_postprocess(self) -> None:
        """Read a configuration folder back, whatever is already loaded."""

        self._collect_vasp_phonon(MODE_BOTH)

    def vasp_phonon_band(self) -> None:
        """Phonon band structure from the collected VASP forces."""

        self._vasp_phonon_figure(MODE_BAND)

    def vasp_phonon_band_with_dos(self) -> None:
        """Band structure and DOS side by side, from VASP forces."""

        self._vasp_phonon_figure(MODE_BOTH)

    def vasp_phonon_projected(self) -> None:
        """Phonon bands with each mode projected onto the elements."""

        self._vasp_phonon_figure(MODE_PROJECTED)

    def vasp_phonon_thermal(self) -> None:
        """Thermal properties from the collected VASP force constants."""

        self._vasp_phonon_figure(MODE_THERMAL)

    def vasp_phonon_visualization(self) -> None:
        """Animate a normal mode from the collected VASP force constants."""

        collected = self._collected_vasp_phonon()

        if collected is None:
            self._collect_vasp_phonon(MODE_BAND, then=self._animate_collected)
            return

        self._animate_collected()

    # ------------------------------------------------------------

    def _collected_vasp_phonon(self):
        """
        The last configuration read back, if there is one.

        Kept on the window rather than on the project because the folder
        is the source of truth here: post-processing is meant to work
        with nothing open, so there may be no project to hang it on.
        """

        from qvista.analyses.phonon_vasp import CollectResult

        result = getattr(self, "_vasp_phonon", None)

        return result if isinstance(result, CollectResult) else None

    def _vasp_phonon_figure(self, mode: str) -> None:
        """Draw one of the four figures, reading the folder if needed."""

        collected = self._collected_vasp_phonon()

        if collected is None:
            self._collect_vasp_phonon(mode)
            return

        # Said every time. These force constants outlive the structure
        # that happens to be open -- that is what makes post-processing
        # a folder someone else ran usable -- so the figure has to name
        # the configuration it came out of rather than let the reader
        # assume it belongs to whatever is on screen.
        if collected.root is not None:
            self.log(f"Drawing from the configuration in {collected.root}.")

        collected.mode = mode

        self._plot_phonon(collected)

    def _animate_collected(self) -> None:
        """Open the mode animation on the collected force constants."""

        collected = self._collected_vasp_phonon()

        if collected is None or collected.phonon_data is None:
            QMessageBox.information(
                self,
                "No force constants",
                "Mode animation needs force constants.\n\n"
                "Run Phonon Analysis > VASP > Post-Processing > Read "
                "Configuration on a finished configuration first.",
            )
            return

        self._animate_modes(collected.phonon_data.phonon)

    def _collect_vasp_phonon(self, mode: str, then=None) -> None:
        """
        Ask for a configuration folder and turn it into force constants.

        ``then`` runs once the result is in hand, which is how Band
        Visualization opens on force constants that did not exist when
        the menu item was clicked.
        """

        from qvista.analyses.phonon_vasp import VaspPhononCollect
        from qvista.gui.dialogs.phonon_vasp_dialog import (
            VaspPostprocessDialog,
        )

        project = self.project_controller.current_project()

        # No project is needed: everything comes from the folder, which
        # is what makes a run done on another machine usable here.
        start = ""

        if project is not None:

            stored = project.results.phonon_vasp.get("configuration")

            if stored is not None:
                start = str(stored.root)

        dialog = VaspPostprocessDialog(start=start, parent=self)

        if dialog.exec() != QDialog.Accepted:
            return

        def done(result: object) -> None:

            self._on_vasp_collect_done(result, mode)

            if then is not None:
                then()

        self._run_analysis(
            VaspPhononCollect(project, **dialog.parameters()),
            on_success=done,
        )

    # ------------------------------------------------------------
    # Electrochemical Tools : Band Structure, DOS, Band Centre
    #
    # Eleven menu entries over three windows. Each entry opens its
    # window with the mode it names already selected; the other modes
    # sit beside it, so moving between them does not mean another trip
    # through the menu.
    # ------------------------------------------------------------

    def band_plot(self) -> None:
        """
        The band-structure window.

        Opens unprojected; atom, element, orbital and sum are the chips
        along the top, which is where the five old menu entries went.
        """

        self._electronic("band", "total")

    def dos_plot(self) -> None:
        """
        The density-of-states window.

        Opens element wise, with atom, orbital, k-point and band
        resolved a click away on the same row.
        """

        self._electronic("dos", "element")

    # The old per-mode entries. Kept as methods because they are a
    # perfectly good way to open the window on one mode, and something
    # else may still call them; they are simply no longer in the menu.

    def projected_band_atom(self) -> None:
        """Bands with each state weighted by chosen atoms."""

        self._electronic("band", "atom")

    def projected_band_element(self) -> None:
        """Bands weighted by species."""

        self._electronic("band", "element")

    def projected_band_orbital(self) -> None:
        """Bands weighted by s, p and d character."""

        self._electronic("band", "orbital")

    def projected_band_sum(self) -> None:
        """Bands weighted by a set of atoms and elements together."""

        self._electronic("band", "sum")

    def projected_dos_atom(self) -> None:
        """Density of states, one curve per atom."""

        self._electronic("dos", "atom")

    def projected_dos_orbital(self) -> None:
        """Density of states, split by shell."""

        self._electronic("dos", "orbital")

    def projected_dos_k_index(self) -> None:
        """Density of states from chosen k-points only."""

        self._electronic("dos", "kpoint")

    def projected_dos_band_index(self) -> None:
        """Density of states from chosen bands only."""

        self._electronic("dos", "band")

    def band_centre(self) -> None:
        """
        First moments of the projected s, p and d states.

        Every shell at once. Which one matters is the question a band
        centre is asked to answer, so picking one first put the
        conclusion before the calculation.
        """

        self._electronic("centre", "orbital")

    # The old per-shell entries. Kept as methods -- they are a perfectly
    # good way in -- but the window computes all three either way.

    def p_band_centre(self) -> None:
        """First moment of the projected p states."""

        self._electronic("centre", "orbital")

    def d_band_centre(self) -> None:
        """First moment of the projected d states -- the d-band model."""

        self._electronic("centre", "orbital")

    def _electronic(self, window: str, mode: str) -> None:
        """
        Open one of the three electronic-structure windows.

        No project is required: everything comes from the run folder,
        which is what makes a calculation done on a cluster usable here.
        The open structure only supplies a starting directory.
        """

        from qvista.gui.dialogs.electronic_dialog import (
            BandCentreDialog,
            BandDialog,
            DosDialog,
        )

        windows = {
            "band": BandDialog,
            "dos": DosDialog,
            "centre": BandCentreDialog,
        }

        project = self.project_controller.current_project()

        start = ""

        if project is not None:

            source = project.info.source_file

            if source is not None:
                start = str(Path(source).parent)

        windows[window](start=start, mode=mode, parent=self).exec()

    # ------------------------------------------------------------
    # Electrochemical Tools : Orbital
    # ------------------------------------------------------------

    def orbital_projected(self) -> None:
        """
        Orbital contributions summed into s, p and d.

        The chemist's reading of a state: "70% O p, 25% Co d".
        """

        self._orbital_dialog(view=ob_service.VIEW_SPD)

    def orbital_resolved(self) -> None:
        """
        The same, with all nine channels kept apart.

        Says *which* d orbital carries the state, which is what a
        crystal-field argument turns on and what summing the shell
        throws away.
        """

        self._orbital_dialog(view=ob_service.VIEW_RESOLVED)

    def orbital_vbm(self) -> None:
        """Orbital contributions and wavefunctions at the valence edge."""

        self._orbital_dialog("vbm")

    def orbital_cbm(self) -> None:
        """The same, at the conduction edge."""

        self._orbital_dialog("cbm")

    def orbital_custom(self) -> None:
        """The same, for a band named by hand."""

        self._orbital_dialog("custom")

    def _orbital_dialog(
        self,
        mode: str = "vbm",
        view: str = ob_service.VIEW_SPD,
    ) -> None:
        """
        Open the Orbital window on one state, at one level of detail.

        No project is required: everything is read from the run folder,
        which is what makes a calculation done on a cluster usable here.
        The open structure only supplies a starting directory.

        Which state -- valence edge, conduction edge, or a band named by
        hand -- is now chosen inside the window rather than by which
        menu entry was clicked, so comparing the two edges does not mean
        closing it and opening it again.
        """

        from qvista.gui.dialogs.orbital_dialog import OrbitalDialog

        project = self.project_controller.current_project()

        start = ""

        if project is not None:

            source = project.info.source_file

            if source is not None:
                start = str(Path(source).parent)

        dialog = OrbitalDialog(
            mode=mode, start=start, view=view, parent=self
        )

        dialog.exec()

    # ------------------------------------------------------------
    # Bonding Tools : Bader
    # ------------------------------------------------------------

    def bader_analysis(self) -> None:
        """
        Partition the charge density into atomic volumes.

        Like the Orbital window this reads a run folder rather than the
        open project, so a calculation done on a cluster can be dropped
        in and analysed without importing it first.
        """

        from qvista.gui.dialogs.bader_dialog import BaderDialog

        project = self.project_controller.current_project()

        start = ""

        if project is not None:

            source = project.info.source_file

            if source is not None:
                start = str(Path(source).parent)

        dialog = BaderDialog(start=start, parent=self)

        dialog.exec()

    def ddec6_analysis(self) -> None:
        """DDEC6 charges and bond orders from a Chargemol run."""

        from qvista.gui.dialogs.ddec_dialog import DdecDialog

        project = self.project_controller.current_project()

        start = ""

        if project is not None:

            source = project.info.source_file

            if source is not None:
                start = str(Path(source).parent)

        dialog = DdecDialog(start=start, parent=self)

        dialog.exec()

    def cohp_analysis(self) -> None:
        """COHP, COOP and COBI bonding analysis from a LOBSTER run."""

        from qvista.gui.dialogs.cohp_dialog import CohpDialog

        project = self.project_controller.current_project()

        start = ""

        if project is not None:

            source = project.info.source_file

            if source is not None:
                start = str(Path(source).parent)

        dialog = CohpDialog(start=start, parent=self)

        dialog.exec()

    # ------------------------------------------------------------

    def _on_vasp_collect_done(
        self, result: object, mode: str = MODE_BOTH
    ) -> None:
        """Cache the force constants and draw the figure that was asked for."""

        from qvista.analyses.phonon_vasp import CollectResult

        if not isinstance(result, CollectResult):
            return

        result.mode = mode

        # Held for the other three figures, which are redraws of this
        # same band, DOS and projection.
        self._vasp_phonon = result

        project = self.project_controller.current_project()

        if project is not None and result.phonon_data is not None:

            # The cache is keyed on the open structure. Claiming these
            # force constants for it when the folder was built from a
            # different cell would let Band Visualization animate one
            # material's modes on another's geometry, so the key is only
            # taken when the two really are the same cell.
            if self._same_cell(project, result.phonon_data.primitive):
                source = project.info.source_file
                result.phonon_data.source = (
                    "" if source is None else str(source)
                )

                self.log(
                    "These force constants are now what Phonon Analysis > "
                    "CHGNet > Band Visualization will animate."
                )

            project.results.phonon = result.as_storage()

        self._plot_phonon(result)

    def _same_cell(self, project: Project, primitive) -> bool:
        """True when the configuration was built from the open structure."""

        import numpy as np

        stored = project.results.kpath.get("result")

        if (
            primitive is None
            or not isinstance(stored, KPathResult)
            or stored.primitive is None
        ):
            return False

        if stored.primitive.natoms != primitive.natoms:
            return False

        drift = np.abs(
            np.asarray(stored.primitive.lattice, dtype=float)
            - np.asarray(primitive.lattice, dtype=float)
        ).max()

        return bool(drift < 1e-3)

    # ------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------

    def _run_analysis(
        self,
        analysis: Analysis,
        on_success=None,
        *,
        verbose: bool = True,
        max_result_lines: int | None = None,
    ) -> None:
        """
        Run an analysis on a worker thread and report it to the console.

        Shared by every analysis menu action -- progress streams into the
        console, the status bar tracks it, and failures become a message
        box rather than a traceback. ``on_success`` receives the
        :class:`~qvista.analyses.base.AnalysisResult` and is where a feature
        stores results or draws a plot.

        ``verbose=False`` drops the run's chrome -- the heading, the
        progress lines and the closing "finished" line -- leaving only the
        result itself. ``max_result_lines`` truncates even that, to the
        first N lines. Neither touches the status bar, which reports the
        run either way. Failures are always reported in full.
        """

        if (
            self._analysis_worker is not None
            and self._analysis_worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "Analysis running",
                f"{self._analysis_worker.analysis.name} is still running. "
                "Wait for it to finish before starting another.",
            )
            return

        self._analysis_verbose = verbose
        self._analysis_max_result_lines = max_result_lines

        if verbose:
            self.log("")
            self.log(analysis.name)
            self.log("----------------")

        self._analysis_on_success = on_success

        worker = AnalysisWorker(analysis, parent=self)

        self._analysis_worker = worker

        worker.progress.connect(self._on_analysis_progress)
        worker.completed.connect(self._on_analysis_completed)
        worker.failed.connect(self._on_analysis_failed)
        worker.cancelled.connect(self._on_analysis_cancelled)

        worker.finished.connect(self._on_analysis_finished)
        worker.finished.connect(worker.deleteLater)

        self.statusBar().showMessage(f"{analysis.name}: starting...")

        worker.start()

    def cancel_analysis(self) -> None:
        """Ask the running analysis, if any, to stop."""

        if (
            self._analysis_worker is not None
            and self._analysis_worker.isRunning()
        ):
            self._analysis_worker.cancel()

            self.statusBar().showMessage("Cancelling...")

    # ------------------------------------------------------------
    # Analysis callbacks
    # ------------------------------------------------------------

    def _analysis_name(self) -> str:
        """Name of the running analysis, for messages."""

        if self._analysis_worker is None:
            return "Analysis"

        return self._analysis_worker.analysis.name

    def _result_lines(self, result: AnalysisResult) -> list[str]:
        """
        The console lines a result contributes, honouring the line cap.

        Notes arrive as multi-line strings, so they are split here: the
        cap counts lines the user actually sees, not blocks.
        """

        lines: list[str] = []

        if result.title:
            lines.append(result.title)

        for label, value in result.as_rows():
            lines.append(f"{label:<19}: {value}")

        for note in result.notes:
            lines.append("")
            lines.extend(note.split("\n"))

        limit = self._analysis_max_result_lines

        if limit is not None:
            lines = lines[:limit]

        return lines

    def _on_analysis_progress(
        self,
        message: str,
        percent: int,
    ) -> None:
        """Stream a progress update to the console and status bar."""

        if self._analysis_verbose:
            self.log(message)

        if percent >= 0:
            self.statusBar().showMessage(
                f"{self._analysis_name()}: {message} ({percent}%)"
            )
        else:
            self.statusBar().showMessage(
                f"{self._analysis_name()}: {message}"
            )

    def _on_analysis_completed(self, result: object) -> None:
        """Print the result, then hand it to the feature's callback."""

        name = self._analysis_name()

        if isinstance(result, AnalysisResult):

            for line in self._result_lines(result):
                self.log(line)

        if self._analysis_verbose:
            self.log("")
            self.log(f"{name} finished.")

        self.statusBar().showMessage(f"{name} finished", 5000)

        if self._analysis_on_success is not None:
            self._analysis_on_success(result)

    def _on_analysis_failed(self, message: str) -> None:
        """Report a failure as a message box, never as a traceback."""

        name = self._analysis_name()

        self.log("")
        self.log(f"{name} failed: {message}")

        self.statusBar().showMessage(f"{name} failed", 5000)

        QMessageBox.critical(self, f"{name} failed", message)

    def _on_analysis_cancelled(self) -> None:
        """Note that the user stopped the run. Not an error."""

        name = self._analysis_name()

        self.log("")
        self.log(f"{name} cancelled.")

        self.statusBar().showMessage(f"{name} cancelled", 5000)

    def _on_analysis_finished(self) -> None:
        """Release the worker once its thread has ended."""

        self._analysis_worker = None
        self._analysis_on_success = None

        # Back to full console output for whatever runs next.
        self._analysis_verbose = True
        self._analysis_max_result_lines = None

    # ------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------

    def update_boundary(self) -> None:
        """
        Update the displayed boundary.
        """

        viewer = self.workspace.viewer
        boundary = viewer.settings.boundary
        panel = self.visualization

        boundary.x_minus = panel.x_minus.value()
        boundary.x_plus = panel.x_plus.value()

        boundary.y_minus = panel.y_minus.value()
        boundary.y_plus = panel.y_plus.value()

        boundary.z_minus = panel.z_minus.value()
        boundary.z_plus = panel.z_plus.value()

        viewer.settings.origin.x = panel.origin_x.value()
        viewer.settings.origin.y = panel.origin_y.value()
        viewer.settings.origin.z = panel.origin_z.value()

        # Refit, or growing the boundary looks like it did nothing.
        #
        # The new copies appear one lattice vector away, and along an
        # axis carrying vacuum that is most of the way across the cell --
        # 30 A for a slab, which lands them far outside the framed view.
        # With no vacuum the copy is adjacent and stays roughly in
        # frame, which is why this only ever looked broken on slabs.
        #
        # reset_camera keeps the view direction and only refits the
        # distance, so the structure does not spin under the cursor.
        viewer.redraw(reset_camera=True)

    def show_about(self) -> None:
        """Display the About dialog."""

        QMessageBox.about(
            self,
            "About QVista",
            (
                f"{APP_NAME} {APP_VERSION}\n\n"
                "Integrated Computational Materials Analysis Platform"
            ),
        )


    # ------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Connect GUI widgets to the viewer."""

        viewer = self.workspace.viewer
        panel = self.visualization



        # ==========================================================
        # Visualization settings
        # ==========================================================

        panel.settingsChanged.connect(
            viewer.apply_settings
        )

        # ==========================================================
        # Boundary
        # ==========================================================

        for widget in (
            panel.x_minus,
            panel.x_plus,
            panel.y_minus,
            panel.y_plus,
            panel.z_minus,
            panel.z_plus,
            # The origin shift goes through the same handler: it changes
            # which images are inside the boundary, so it needs the same
            # rebuild and the same refit.
            panel.origin_x,
            panel.origin_y,
            panel.origin_z,
        ):
            widget.valueChanged.connect(
                self.update_boundary
            )

        # ==========================================================
        # Camera buttons
        # ==========================================================

        panel.x_button.clicked.connect(
            viewer.view_x
        )

        panel.minus_x_button.clicked.connect(
            viewer.view_minus_x
        )

        panel.y_button.clicked.connect(
            viewer.view_y
        )

        panel.minus_y_button.clicked.connect(
            viewer.view_minus_y
        )

        panel.z_button.clicked.connect(
            viewer.view_z
        )

        panel.minus_z_button.clicked.connect(
            viewer.view_minus_z
        )

        panel.iso_button.clicked.connect(
            viewer.view_iso
        )

        panel.reset_button.clicked.connect(
            viewer.reset_view
        )

        panel.drag_button.toggled.connect(
            viewer.set_drag_mode
        )

        # ==========================================================
        # Bond types
        # ==========================================================

        panel.bond_types_button.clicked.connect(
            self.bond_types
        )

    # ------------------------------------------------------------
    # Bond types
    # ------------------------------------------------------------

    def bond_types(self) -> None:
        """
        Choose which element pairs are drawn as bonded.

        The pairs are taken from the structure on screen rather than the
        project, since that is what the rules will be judged against.
        With nothing open the dialog still opens and says so -- there is
        simply nothing to make a rule about.
        """

        from qvista.gui.dialogs.bond_dialog import BondDialog

        viewer = self.workspace.viewer
        panel = self.visualization

        elements: list[str] = []

        if viewer.structure is not None:

            seen: list[str] = []

            for symbol in viewer.structure.atoms.get_chemical_symbols():

                if symbol not in seen:
                    seen.append(symbol)

            elements = seen

        dialog = BondDialog(
            elements, panel.bond_rules(), parent=self
        )

        # Live, not on OK: the question being answered is whether the
        # picture reads correctly, which cannot be judged from a dialog
        # that hides the picture until it is dismissed.
        dialog.rulesChanged.connect(panel.set_bond_rules)

        dialog.exec()



