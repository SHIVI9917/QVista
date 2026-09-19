"""
Electronegativity panel.

Non-modal on purpose: it stays open while the user clicks around the
structure, and rewrites itself on every selection change. Clicking one
atom describes that site against the cell it sits in; clicking a second
describes the pair.

Picking is armed only while this dialog is open. Left-click is also how
the camera is rotated, so leaving it armed would mean selecting an atom
every time the view is turned.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from qvista.gui import theme
from qvista.gui.sizing import fit_to_screen
from qvista.gui.dialogs.kpath_dialog import (
    INFO_BUTTON_STYLE,
    _hline,
)
from qvista.services.electronegativity import (
    Scale,
    cell_electronegativity,
    chi,
    compare,
    polyhedron,
)
from qvista.visualization.coordination import (
    boundary_key,
    coordination_shell,
    neighbour_pool,
)
from qvista.visualization.selection import atom_key
from qvista.visualization.viewer import SELECTION_COLORS

#: Monospace, so the value columns line up.
REPORT_STYLE = theme.MONO_STYLE

#: Neighbours listed for a single selected atom. A dense structure can
#: have a dozen; the nearest few are the ones that matter.
MAX_NEIGHBOURS = 8


def _format_value(value: float | None, digits: int = 2) -> str:
    """A number, or a dash where there is none."""

    return "--" if value is None else f"{value:.{digits}f}"


class ElectronegativityDialog(QDialog):
    """Live electronegativity readout for the current selection."""

    def __init__(self, viewer, parent=None, polyhedral: bool = False) -> None:

        super().__init__(parent)

        self.viewer = viewer

        #: Padded neighbour pool, and the structure/boundary it is for.
        self._pool: list = []
        self._pool_key = None

        self.setWindowTitle("Electronegativity")
        fit_to_screen(self, 640, 480)

        # Non-modal: the point is to keep clicking the structure behind
        # it. Qt.Tool keeps it above the main window without stealing
        # focus from the viewport.
        self.setModal(False)
        self.setWindowFlag(Qt.Tool, True)

        layout = QVBoxLayout(self)

        first, second = SELECTION_COLORS[:2]

        instructions = QLabel(
            "Click an atom to describe it against the cell; click a "
            "second to compare the pair.\n"
            f"Cages mark atom 1 ({first}) and atom 2 ({second}). "
            "Click again to deselect, right-click to clear.\n"
            "Dragging still rotates the view."
        )
        instructions.setWordWrap(True)

        layout.addWidget(instructions)

        # The coordination shell as a whole, rather than one bond at a
        # time. Off by default: the pair comparison is the common case.
        self.polyhedral_box = QCheckBox(
            "Polyhedral environment -- sum over the whole coordination "
            "shell of the selected atom"
        )
        self.polyhedral_box.setChecked(polyhedral)
        self.polyhedral_box.toggled.connect(self._on_mode_changed)

        layout.addWidget(self.polyhedral_box)

        layout.addWidget(_hline())

        self.report = QLabel()
        self.report.setStyleSheet(REPORT_STYLE)
        self.report.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.report.setWordWrap(True)

        layout.addWidget(self.report, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        clear_button = QPushButton("Clear selection")
        clear_button.setStyleSheet(INFO_BUTTON_STYLE)
        clear_button.clicked.connect(self._clear)

        buttons.addWidget(clear_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        buttons.addWidget(close_button)

        layout.addLayout(buttons)

        # ----------------------------------------------------------

        self.viewer.enable_atom_picking(self.on_selection_changed)

        self.on_selection_changed(self.viewer.selection.atoms)

    # ------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Disarm picking, so ordinary rotation stops selecting atoms."""

        self.viewer.disable_atom_picking()

        super().closeEvent(event)

    def _clear(self) -> None:

        self.viewer.clear_selection()

    def set_polyhedral(self, polyhedral: bool) -> None:
        """
        Switch between the pair comparison and the coordination shell.

        For the menu, which has an entry per view and one panel to put
        them in: picking the other entry while this is already open has
        to move the panel onto that view. Setting the box is enough --
        it is what the report reads, and toggling it redraws.
        """

        self.polyhedral_box.setChecked(bool(polyhedral))

    def _on_mode_changed(self, _checked: bool) -> None:
        """Redraw the report without touching the selection."""

        self.on_selection_changed(self.viewer.selection.atoms)

    # ------------------------------------------------------------

    def on_selection_changed(self, atoms) -> None:
        """Rewrite the report for the current selection."""

        self.report.setText("\n".join(self._report(atoms)))

    # ------------------------------------------------------------

    def _report(self, atoms) -> list[str]:
        """The whole readout, as lines."""

        if self.viewer.structure is None:
            return ["No structure loaded."]

        lines = list(self._selection_lines(atoms))

        if len(atoms) >= 2:
            lines += [""] + self._pair_lines(atoms[-2], atoms[-1])

        elif len(atoms) == 1:
            lines += [""] + self._site_lines(atoms[0])

        lines += [""] + self._cell_lines()

        return lines

    def _selection_lines(self, atoms) -> list[str]:
        """Which atoms are selected, and where they are."""

        if not atoms:
            return ["Nothing selected -- click an atom."]

        lines = ["Selected"]

        for index, atom in enumerate(atoms, start=1):

            site, tx, ty, tz = atom_key(atom)

            image = (
                "home cell"
                if (tx, ty, tz) == (0, 0, 0)
                else f"image ({tx:+d}, {ty:+d}, {tz:+d})"
            )

            lines.append(
                f"  {index}   {atom.symbol:<3}  site {site:<4} {image}"
            )

        return lines

    # ------------------------------------------------------------

    def _pair_lines(self, first, second) -> list[str]:
        """The comparison between two selected atoms."""

        distance = float(
            ((first.position - second.position) ** 2).sum() ** 0.5
        )

        result = compare(
            first.symbol,
            second.symbol,
            distance=distance,
            bonded=self._are_bonded(first, second),
        )

        if not result.defined:
            return [
                f"{first.symbol} and {second.symbol} have no "
                f"electronegativity on either the Pauling or the "
                f"Mulliken scale."
            ]

        unit = f" {result.scale.unit}" if result.scale.unit else ""

        lines = [
            f"Pair  ({result.scale} scale)",
            f"  chi({result.symbol_a})".ljust(18)
            + f"{result.chi_a:.2f}{unit}",
            f"  chi({result.symbol_b})".ljust(18)
            + f"{result.chi_b:.2f}{unit}",
            "  delta chi".ljust(18) + f"{result.delta:.2f}{unit}",
        ]

        toward = result.more_electronegative

        if toward is not None:
            lines.append(
                "  direction".ljust(18)
                + f"electron density towards {toward}"
            )
        else:
            lines.append("  direction".ljust(18) + "equal, no polarity")

        if result.ionic_percent is not None:
            lines.append(
                "  ionic character".ljust(18)
                + f"{result.ionic_percent:.1f} %   ({result.character})"
            )
        else:
            # Pauling's relation is fitted to Pauling differences; an eV
            # difference put through it would be a meaningless number.
            lines.append(
                "  ionic character".ljust(18)
                + "not defined on the Mulliken scale"
            )

        lines.append(
            "  separation".ljust(18)
            + f"{distance:.3f} A"
            + ("   bonded" if result.bonded else "   not bonded")
        )

        return lines

    def _are_bonded(self, first, second) -> bool:
        """Whether the viewer's own bonding rules join these two atoms."""

        for bond in self._bonds():

            ends = {atom_key(bond.atom_i), atom_key(bond.atom_j)}

            if ends == {atom_key(first), atom_key(second)}:
                return True

        return False

    def _bonds(self) -> list:
        """Bonds in the current scene, or an empty list."""

        scene_data = getattr(self.viewer.scene, "scene_data", None)

        if scene_data is None or scene_data.graph is None:
            return []

        return scene_data.graph.bonds

    # ------------------------------------------------------------

    def _site_lines(self, atom) -> list[str]:
        """One atom against its cell, plus its bonded neighbours."""

        cell = self._cell_result()

        lines: list[str] = []

        if cell.defined:

            value = cell.deviation(atom.symbol)

            unit = f" {cell.scale.unit}" if cell.scale.unit else ""

            lines += [
                f"Site  ({cell.scale} scale)",
                f"  chi({atom.symbol})".ljust(18)
                + _format_value(chi(atom.symbol, cell.scale))
                + unit,
                "  cell chi_eq".ljust(18)
                + f"{cell.equalized:.2f}{unit}",
            ]

            if value is not None:

                direction = (
                    "pulls electron density in"
                    if value > 0
                    else "gives electron density up"
                )

                lines.append(
                    "  deviation".ljust(18)
                    + f"{value:+.2f}{unit}   ({atom.symbol} {direction})"
                )

        if self.polyhedral_box.isChecked():
            extra = self._polyhedron_lines(atom)
        else:
            extra = self._neighbour_lines(atom)

        if extra:
            lines += [""] + extra

        return lines

    # ------------------------------------------------------------

    def _neighbour_pool(self):
        """The padded pool for the current structure, built once."""

        boundary = self.viewer.settings.boundary

        key = boundary_key(self.viewer.structure, boundary)

        if self._pool_key != key:

            self._pool = neighbour_pool(
                self.viewer.scene, self.viewer.structure, boundary
            )
            self._pool_key = key

        return self._pool

    def _coordination(self, atom):
        """
        The bonded shell around ``atom``.

        The polyhedron builder is deliberately not used: it only accepts
        cations as centres and only runs while polyhedra are displayed,
        and this has to work for any atom the user clicks.
        """

        return coordination_shell(
            atom,
            self._neighbour_pool(),
            self.viewer.scene.visible_bond_builder.scale,
        )

    def _polyhedron_lines(self, atom) -> list[str]:
        """What the whole coordination shell adds up to."""

        shell = self._coordination(atom)

        if not shell:
            return [
                "Polyhedral environment",
                "  nothing bonded to this atom in view",
            ]

        result = polyhedron(atom.symbol, shell)

        if not result.defined:
            return [
                "Polyhedral environment",
                "  no electronegativity scale covers this shell",
            ]

        unit = f" {result.scale.unit}" if result.scale.unit else ""

        lines = [
            f"Polyhedral environment  ({result.scale} scale)",
            "  coordination".ljust(18)
            + f"{result.coordination}   ({result.composition})",
            "  chi(centre)".ljust(18)
            + f"{result.chi_centre:.2f}{unit}   {result.centre}",
            "  mean ligand chi".ljust(18)
            + f"{result.mean_ligand_chi:.2f}{unit}   (geometric)",
            "  net delta chi".ljust(18)
            + f"{result.net_delta:+.2f}{unit}   "
            + (
                f"the shell pulls density off {result.centre}"
                if result.net_delta > 0
                else f"the shell gives density to {result.centre}"
            ),
        ]

        # The vector sum is the part a per-bond list cannot show: six
        # equal pulls in opposite directions cancel, the same six to one
        # side do not.
        lines += [
            "  resultant".ljust(18)
            + f"{result.resultant:.2f}{unit}"
            + f"   of {result.scalar_total:.2f} if aligned",
            "  asymmetry".ljust(18)
            + f"{result.asymmetry:.2f}   ({result.balance})",
        ]

        # Cation-cation contacts are bonded by the drawing rule but are
        # not what people mean by the coordination polyhedron, and they
        # pull the mean towards the centre's own value.
        attracting = result.attracting()

        if (
            attracting.defined
            and attracting.coordination != result.coordination
        ):
            lines += [
                "",
                f"  Counting only the {attracting.coordination} "
                f"neighbours that pull density away "
                f"({attracting.composition}):",
                "    mean ligand chi".ljust(22)
                + f"{attracting.mean_ligand_chi:.2f}{unit}",
                "    net delta chi".ljust(22)
                + f"{attracting.net_delta:+.2f}{unit}",
                "    asymmetry".ljust(22)
                + f"{attracting.asymmetry:.2f}"
                + f"   ({attracting.balance})",
            ]

        lines += ["", "  Shell"]

        for ligand in result.ligands[:MAX_NEIGHBOURS]:

            lines.append(
                f"    {ligand.symbol:<3} {ligand.distance:7.3f} A"
                f"   delta {ligand.delta:+.2f}{unit}"
            )

        if result.coordination > MAX_NEIGHBOURS:
            lines.append(
                f"    ... and {result.coordination - MAX_NEIGHBOURS} more"
            )

        return lines

    def _neighbour_lines(self, atom) -> list[str]:
        """Δχ from this atom to each atom it is bonded to."""

        found = sorted(
            (
                (distance, symbol)
                for symbol, distance, _ in self._coordination(atom)
            )
        )

        if not found:
            return ["Bonded neighbours", "  nothing bonded to this atom"]

        lines = [f"Bonded neighbours ({len(found)})"]

        for distance, symbol in found[:MAX_NEIGHBOURS]:

            result = compare(atom.symbol, symbol)

            delta = _format_value(result.delta)

            percent = (
                f"{result.ionic_percent:.0f} % ionic"
                if result.ionic_percent is not None
                else f"{result.scale} scale" if result.scale else "--"
            )

            lines.append(
                f"  {symbol:<3} {distance:7.3f} A"
                f"   delta {delta:>5}   {percent}"
            )

        if len(found) > MAX_NEIGHBOURS:
            lines.append(f"  ... and {len(found) - MAX_NEIGHBOURS} more")

        return lines

    # ------------------------------------------------------------

    def _cell_result(self):
        """Sanderson mean over the atoms of the loaded unit cell."""

        return cell_electronegativity(
            self.viewer.structure.atoms.get_chemical_symbols()
        )

    def _cell_lines(self) -> list[str]:
        """The cell-wide number, shown whatever is selected."""

        cell = self._cell_result()

        if not cell.defined:
            return [
                "Cell",
                "  no electronegativity scale covers every element here",
            ]

        unit = f" {cell.scale.unit}" if cell.scale.unit else ""

        note = (
            ""
            if cell.scale is Scale.PAULING
            else "   (Pauling undefined for one of these elements)"
        )

        return [
            f"Cell  (Sanderson equalization, {cell.scale} scale)",
            "  chi_eq".ljust(18)
            + f"{cell.equalized:.3f}{unit}"
            + f"   over {len(cell.symbols)} atoms, "
            + self.viewer.structure.formula
            + note,
        ]
