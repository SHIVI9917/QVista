"""
Crystal viewer based on PyVistaQt.
"""

from __future__ import annotations

import pyvista as pv
import vtk
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

from qvista.gui import theme
from qvista.models.structure import Structure
from qvista.visualization.navigation import (
    enable_zoom_to_cursor,
    mute_style_wheel,
)
from qvista.visualization.radii import covalent_radius, vdw_radius
from qvista.visualization.render_cache import RenderCache
from qvista.visualization.scene import Scene, report_timing
from qvista.visualization.selection import AtomSelection, nearest_atom
from qvista.visualization.settings import (
    Projection,
    Representation,
    ViewerSettings,
)

#: Highlight colour per clicked atom, in click order. One colour each
#: rather than one for all, so a panel can say which atom is which --
#: and for a bond angle, which of the three is the corner.
SELECTION_COLORS = ("#ffcc00", "#00e5ff", "#ff6ec7")

#: Highlight cage size, relative to the drawn sphere.
SELECTION_SCALE = 1.45

#: How far a press may travel and still count as a click rather than a
#: drag. Without this, every rotate would select whatever was under the
#: cursor when the drag started.
CLICK_TOLERANCE_PX = 4

#: Shown while nothing is loaded, and taken down as soon as something is.
HINT_TEXT = "No structure loaded\n\nFile  >  Open Project    (Ctrl+O)"

#: The theme's blue, so the one thing on an empty viewport belongs to
#: the application around it. Iris deep rather than iris: the lighter
#: one is pitched for a dark surface and goes weak on white.
HINT_COLOR = theme.IRIS_DEEP

#: The x/y/z letters on the orientation marker. Dark, now that the
#: viewport is white again.
AXES_LABEL_COLOR = "#3c4a68"


def _only_projection_changed(
    before: ViewerSettings,
    after: ViewerSettings,
) -> bool:
    """
    Whether the two settings differ in the projection and nothing else.

    The panel rebuilds its settings from scratch on every change and
    emits the lot, so the only way to know that clicking "Parallel" did
    not also change something that needs geometry is to compare. Every
    field except the projection has to match, which is why this walks
    the dataclass rather than naming the ones that matter -- a field
    added later would otherwise be silently exempt from the check and
    stop taking effect.
    """

    if before.projection is after.projection:
        return False

    return all(
        getattr(before, name) == getattr(after, name)
        for name in ViewerSettings.__slots__
        if name != "projection"
    )


class CrystalViewer(QtInteractor):
    """Embedded PyVista crystal viewer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # ---------------------------------------------------------
        # Selection
        # ---------------------------------------------------------

        self.selection = AtomSelection()

        #: Called after any selection change, while picking is on.
        self._selection_callback = None

        #: How many highlight cages were drawn last time. Kept because
        #: the cages are named actors and the ones no longer wanted have
        #: to be named to be removed -- a tool that allows a hundred
        #: picks and then drops to two would otherwise leave the other
        #: ninety-eight on screen.
        self._selection_slots = 0

        self._picking_enabled = False
        self._press_position = None
        self._pick_observers: list = []

        # ---------------------------------------------------------
        # Data
        # ---------------------------------------------------------

        self.structure: Structure | None = None

        self.settings = ViewerSettings()

        #: False while something is drawing its own repeated cell and
        #: does not want the boundary tiling on top of it. A mode
        #: animation is the case: it builds the repetition itself, with
        #: the phase each cell is supposed to carry.
        self.tile_images = True

        #: True while the left button slides the scene rather than
        #: turning it. Not a :class:`ViewerSettings` field: it changes
        #: how the mouse is read, not what is drawn, and putting it there
        #: would make every drag toggle rebuild the geometry.
        self.drag_mode = False

        # ---------------------------------------------------------
        # Viewer appearance
        # ---------------------------------------------------------

        self.set_background(
            self.settings.background_color,
            top=self.settings.background_top,
        )

        #: The line drawn across an empty viewport. See :meth:`_update_hint`.
        self._hint_actor = None

        # The wheel zooms towards the pointer, not the middle of the
        # view. An atom worth looking at closely is off-centre, and the
        # default would push it out of frame.
        enable_zoom_to_cursor(self)

        if self.settings.show_axes:
            self.cache = RenderCache()
            self.cache.axes_actor = self.add_axes(color=AXES_LABEL_COLOR)
        else:
            self.cache = RenderCache()

        self._update_hint()

        # Down the c axis in parallel projection, which is the view a
        # structure is usually opened to be checked in: layers stack
        # along c far more often than not, and it is the one direction
        # where "is this channel open" has a yes-or-no answer on screen.
        # Both are settings, so a click on Iso or Perspective moves off
        # it and stays moved.
        self.apply_projection()

        self.view_xy()

        # ---------------------------------------------------------
        # Scene
        # ---------------------------------------------------------

        self.scene = Scene()

    # =============================================================
    # Structure
    # =============================================================

    def load_structure(
        self,
        structure: Structure,
    ) -> None:
        """
        Load a structure.
        """

        self.structure = structure

        # A selection names sites in the old structure, which mean
        # something different -- or nothing -- in the new one. The same
        # goes for a frozen bond list.
        self.clear_selection(notify=False)

        self.unfreeze_bonds()

        self.redraw(
            reset_camera=True,
        )

        # Whatever is showing the numbers is now describing a structure
        # that is no longer loaded.
        self._notify_selection()

    def freeze_bonds(self) -> None:
        """
        Keep the bonds that are drawn now, wherever the atoms go next.

        For animation. Bonds are decided by distance, and a vibrating
        atom crosses the cutoff twice a period -- so without this the
        structure visibly comes apart and reassembles as it oscillates,
        which reads as the molecule breaking rather than the bond
        stretching.
        """

        from qvista.visualization.visible_bond_builder import bond_pairs

        if self.scene.scene_data is None or self.scene.scene_data.graph is None:
            return

        self.scene.fixed_bonds = bond_pairs(self.scene.scene_data.graph)

    def unfreeze_bonds(self) -> None:
        """Go back to deciding bonds by distance on every redraw."""

        self.scene.fixed_bonds = None

    def update_positions(self, positions) -> None:
        """
        Move the atoms of the loaded structure, in place.

        For a geometry edit rather than a new file: the camera stays
        where it is and the selection survives, so a bond can be nudged
        and looked at without losing the view or having to click the same
        atoms again. Bonds are rebuilt from the new positions unless
        :meth:`freeze_bonds` has been called.
        """

        import numpy as np

        if self.structure is None:
            return

        self.structure.atoms.set_positions(
            np.asarray(positions, dtype=float)
        )

        self.redraw(reset_camera=False)

    # =============================================================
    # Settings
    # =============================================================

    def apply_settings(
        self,
        settings: ViewerSettings,
    ) -> None:
        """
        Apply visualization settings.

        (For now we still redraw. In the next step this
        will intelligently choose between redraw() and
        update_visibility().)
        """

        previous = self.settings

        self.settings = settings

        self.apply_projection()

        # The projection is the camera's business, not the scene's. A
        # change to it alone -- which is most of what the two buttons
        # do -- costs a render rather than a rebuild of every sphere and
        # cylinder in the cell.
        if _only_projection_changed(previous, settings):
            self.render()
            return

        self.redraw(
            reset_camera=False,
        )

    # =============================================================
    # Drag mode
    # =============================================================

    def set_drag_mode(self, on: bool) -> None:
        """
        Make the left button slide the scene instead of turning it.

        Panning is already on the middle button and on shift-left, but
        neither is reachable on a trackpad, and sliding a structure
        across the viewport to look at a different part of it is
        something done constantly while a figure is being framed. So it
        gets a mode of its own.

        Only the button mapping changes. Nothing moves in the structure
        and nothing is redrawn -- this is the camera sliding sideways,
        which is why it costs a style swap and no geometry.
        """

        self.drag_mode = bool(on)

        if self.drag_mode:
            self.enable_custom_trackball_style(left="pan")

        else:
            self.enable_custom_trackball_style()

        # A fresh style arrives at VTK's default wheel factor, and the
        # wheel would then zoom twice per notch -- once from the style
        # and once from the zoom-to-cursor observers, which are on the
        # interactor and survived the swap.
        mute_style_wheel(self)

        self.setCursor(
            Qt.OpenHandCursor if self.drag_mode else Qt.ArrowCursor
        )

    # =============================================================
    # Projection
    # =============================================================

    def apply_projection(self) -> None:
        """
        Put the camera into perspective or parallel projection.

        VTK keeps a separate scale for the parallel camera, and it is
        only meaningful once something has been framed. Resetting the
        clipping range after the switch is what stops the structure
        being sliced open by a near plane computed for the other
        projection.
        """

        if self.settings.projection is Projection.PARALLEL:
            self.enable_parallel_projection()

        else:
            self.disable_parallel_projection()

        self.reset_camera_clipping_range()

    # =============================================================
    # Fast visibility update
    # =============================================================

    def update_visibility(self) -> None:
        """
        Toggle visibility without rebuilding geometry.
        """

        self.cache.update_visibility(
            self.settings,
        )

        # -------------------------------
        # Axes
        # -------------------------------

        if self.settings.show_axes:

            if self.cache.axes_actor is None:

                self.cache.axes_actor = self.add_axes(color=AXES_LABEL_COLOR)

        else:

            if self.cache.axes_actor is not None:

                try:
                    self.hide_axes()
                except Exception:
                    pass

                self.cache.axes_actor = None

        self.render()

    # =============================================================
    # Empty state
    # =============================================================

    def _update_hint(self) -> None:
        """
        Say what to do while there is nothing to look at.

        An empty 3D viewport is indistinguishable from a broken one.
        Centred rather than tucked in a corner, because the point is
        that it is the only thing on screen.
        """

        if self.structure is not None:

            if self._hint_actor is not None:
                self.remove_actor(self._hint_actor, render=False)
                self._hint_actor = None

            return

        if self._hint_actor is not None:
            return

        self._hint_actor = self.add_text(
            HINT_TEXT,
            position=(0.5, 0.5),
            font_size=13,
            color=HINT_COLOR,
            viewport=True,
            name="empty_hint",
        )

        # add_text anchors at the lower left of the string, so a
        # position of (0.5, 0.5) puts the text beside the centre rather
        # than on it until the justification is changed.
        self._hint_actor.GetTextProperty().SetJustificationToCentered()

    # =============================================================
    # Full redraw
    # =============================================================

    def redraw(
        self,
        reset_camera: bool = False,
    ) -> None:
        """
        Completely rebuild the scene.
        """


        import time

        t0 = time.perf_counter()
        # Remove scene actors

        # ---------------------------------------------------------
        # Geometry update
        # ---------------------------------------------------------

        if self.structure is not None:

            self.scene.update_geometry(
                self,
                self.structure,
            )
        # Background

        self.set_background(
            self.settings.background_color,
            top=self.settings.background_top,
        )

        # Axes

        if self.settings.show_axes:

            if self.cache.axes_actor is None:

                self.cache.axes_actor = self.add_axes(color=AXES_LABEL_COLOR)

        else:

            if self.cache.axes_actor is not None:

                try:
                    self.hide_axes()
                except Exception:
                    pass

                self.cache.axes_actor = None

        # Nothing loaded

        self._update_hint()

        if self.structure is None:

            self.render()

            return


        # ---------------------------------------------------------
        # Camera
        # ---------------------------------------------------------

        if reset_camera:

            self.reset_camera()

        # ---------------------------------------------------------
        # Selection
        #
        # The scene was rebuilt from scratch, so the selected atoms are
        # stale objects and their highlights are stale actors. Re-resolve
        # both against what is now on screen.
        # ---------------------------------------------------------

        self._refresh_selection()

        # ---------------------------------------------------------
        # Measure VTK render time
        # ---------------------------------------------------------

        t_render = time.perf_counter()

        self.render()

        report_timing("VTK render", t_render)

        report_timing("Redraw", t0)
    # =============================================================
    # Atom picking
    # =============================================================

    def enable_atom_picking(self, callback=None, limit: int = 2) -> None:
        """
        Let the user select atoms by clicking them.

        ``callback`` is called with the selection list after every change.
        ``limit`` is how many atoms the asking tool wants at once -- two
        for a distance, three for an angle.

        Picking is only armed while a tool wants it: left-clicking is
        also how the camera is rotated, so leaving it on permanently
        would mean selecting an atom every time the view is turned.
        """

        self._selection_callback = callback

        self.set_selection_limit(limit)

        if self._picking_enabled:
            return

        # The press is recorded and the pick happens on release, so a
        # drag can be told apart from a click.
        self._pick_observers = [
            self.iren.add_observer(
                "LeftButtonPressEvent", self._on_left_press
            ),
            self.iren.add_observer(
                "LeftButtonReleaseEvent", self._on_left_release
            ),
            self.iren.add_observer(
                "RightButtonPressEvent", self._on_right_press
            ),
        ]

        self._picking_enabled = True

    def disable_atom_picking(self) -> None:
        """Stop picking, and drop whatever was selected."""

        for observer in self._pick_observers:

            try:
                self.iren.remove_observer(observer)
            except Exception:
                pass

        self._pick_observers = []
        self._picking_enabled = False
        self._selection_callback = None

        self.selection.limit = 2

        self.clear_selection(notify=False)

    def set_selection_limit(self, limit: int) -> None:
        """
        Change how many atoms may be selected at once.

        Shrinking the cap drops the oldest picks, so the highlights and
        whatever is reading them have to be brought back into line.
        """

        if limit == self.selection.limit:
            return

        before = len(self.selection)

        self.selection.limit = limit

        self._draw_selection()
        self.render()

        if len(self.selection) != before:
            self._notify_selection()

    # -------------------------------------------------------------

    def _on_left_press(self, *_args) -> None:
        """Remember where a press started."""

        self._press_position = self.iren.get_event_position()

    def _on_left_release(self, *_args) -> None:
        """Pick, unless the press turned into a drag."""

        start = self._press_position

        self._press_position = None

        if start is None:
            return

        end = self.iren.get_event_position()

        if (
            abs(end[0] - start[0]) > CLICK_TOLERANCE_PX
            or abs(end[1] - start[1]) > CLICK_TOLERANCE_PX
        ):
            return

        self._pick_at(end[0], end[1])

    def _on_right_press(self, *_args) -> None:
        """Right-click clears the selection."""

        if len(self.selection):
            self.clear_selection()

    # -------------------------------------------------------------

    def _pick_at(self, x: int, y: int) -> None:
        """Resolve a click at display coordinates to an atom."""

        if self.scene.scene_data is None:
            return

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0025)
        picker.Pick(x, y, 0, self.renderer)

        actor = picker.GetActor()

        if actor is None:
            return

        symbol = self._symbol_for_actor(actor)

        # Clicking a bond, the cell box or the background is not a
        # selection, and must not be silently rounded to a nearby atom.
        if symbol is None:
            return

        atom = nearest_atom(
            self.scene.scene_data.visible_atoms,
            picker.GetPickPosition(),
            symbol=symbol,
        )

        if atom is None:
            return

        self.selection.pick(atom)

        self._draw_selection()
        self.render()

        self._notify_selection()

    def _symbol_for_actor(self, actor) -> str | None:
        """
        Which element's mesh was hit, or ``None`` if it was not an atom.

        Atoms are drawn one merged mesh per element, so the actor
        identifies the element and nothing finer -- which is exactly what
        the nearest-centre lookup needs to disambiguate the click.
        """

        for name, candidate in self.cache.atom_actors.items():

            if candidate is actor:
                return name.split("atoms_", 1)[-1]

        return None

    # -------------------------------------------------------------

    def select_index(self, index: int) -> bool:
        """
        Highlight one atom of the loaded structure, by its index.

        For a tool that already knows which atom matters and should not
        make the user hunt for it -- the migrating atom of a NEB, say.
        The home cell's copy is chosen rather than whichever periodic
        image comes first, so the highlight lands where the structure
        was drawn from.

        False when there is nothing to select, which is not an error:
        the boundary may have trimmed the atom out of the view.
        """

        data = self.scene.scene_data

        if data is None or not data.visible_atoms:
            return False

        wanted = [
            atom
            for atom in data.visible_atoms
            if atom.original_index == index
        ]

        if not wanted:
            return False

        home = [atom for atom in wanted if atom.is_original_cell]

        self.selection.clear()
        self.selection.pick(home[0] if home else wanted[0])

        self._draw_selection()

        self.render()

        return True

    def clear_selection(self, notify: bool = True) -> None:
        """Drop the selection and its highlights."""

        self.selection.clear()

        self._draw_selection()

        self.render()

        if notify:
            self._notify_selection()

    def _refresh_selection(self) -> None:
        """Re-resolve the selection after the scene was rebuilt."""

        if not len(self.selection):
            self._draw_selection()
            return

        atoms = (
            self.scene.scene_data.visible_atoms
            if self.scene.scene_data is not None
            else []
        )

        changed = self.selection.resync(atoms)

        self._draw_selection()

        if changed:
            self._notify_selection()

    def _draw_selection(self) -> None:
        """Put a cage around each selected atom."""

        # Every slot drawn last time is cleared, not just the ones
        # currently filled: the selection can shrink between calls, and
        # a cage left behind marks an atom that is no longer selected.
        for index in range(max(self._selection_slots, AtomSelection.MAX)):

            try:
                self.remove_actor(f"selection_{index}", render=False)
            except Exception:
                pass

        self._selection_slots = len(self.selection.atoms)

        for index, atom in enumerate(self.selection.atoms):

            sphere = pv.Sphere(
                radius=SELECTION_SCALE * self._drawn_radius(atom.symbol),
                center=atom.position,
                theta_resolution=24,
                phi_resolution=24,
            )

            self.add_mesh(
                sphere,
                style="wireframe",
                line_width=2,
                color=SELECTION_COLORS[index % len(SELECTION_COLORS)],
                name=f"selection_{index}",
                # The cage sits over the atom; a pickable cage would
                # swallow the click that deselects it.
                pickable=False,
                render=False,
            )

    def _drawn_radius(self, symbol: str) -> float:
        """The radius the atom renderer used for this element."""

        if self.settings.representation == Representation.SPACE_FILL:
            base = vdw_radius(symbol)
        else:
            base = covalent_radius(symbol)

        return self.settings.atom_radius_scale * base

    def _notify_selection(self) -> None:
        """Tell whoever asked for picking that the selection moved."""

        if self._selection_callback is not None:
            self._selection_callback(self.selection.atoms)

    # =============================================================
    # Camera helper
    # =============================================================

    def _update_camera(self) -> None:

        self.reset_camera_clipping_range()

        self.render()

    # =============================================================
    # Standard views
    # =============================================================

    def view_x(self):

        self.view_yz()

        self._update_camera()

    def view_minus_x(self):

        self.view_yz()

        self.camera.Azimuth(180)

        self._update_camera()

    def view_y(self):

        self.view_xz()

        self._update_camera()

    def view_minus_y(self):

        self.view_xz()

        self.camera.Azimuth(180)

        self._update_camera()

    def view_z(self):

        self.view_xy()

        self._update_camera()

    def view_minus_z(self):

        self.view_xy()

        self.camera.Azimuth(180)

        self._update_camera()

    def view_iso(self):

        self.view_isometric()

        self.reset_camera()

        self.render()

    def reset_view(self):

        self.view_isometric()

        self.reset_camera()

        self.render()

