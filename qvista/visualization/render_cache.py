"""
Render cache.

Stores actor handles so the viewer does not need to clear and
recreate the entire scene every redraw.
"""

from __future__ import annotations


class RenderCache:
    """
    Cache of all rendered actors.
    """

    def __init__(self) -> None:

        self.clear()

    # ==============================================================
    # Reset
    # ==============================================================

    def clear(self) -> None:

        self.atom_actors = {}

        self.bond_actors = {}

        self.polyhedron_actors = {}

        self.label_actors = {}

        self.cell_actor = None

        self.axes_actor = None

    # ==============================================================
    # Atoms
    # ==============================================================

    def add_atom_actor(
        self,
        name,
        actor,
    ):

        self.atom_actors[name] = actor

    def remove_atoms(
        self,
        viewer,
    ):

        for actor in self.atom_actors.values():

            try:
                viewer.remove_actor(actor, render=False)
            except Exception:
                pass

        self.atom_actors.clear()

    def set_atoms_visible(
        self,
        visible: bool,
    ) -> None:

        for actor in self.atom_actors.values():

            actor.SetVisibility(visible)

    # ==============================================================
    # Bonds
    # ==============================================================

    def add_bond_actor(
        self,
        name,
        actor,
    ):

        self.bond_actors[name] = actor

    def remove_bonds(
        self,
        viewer,
    ):

        for actor in self.bond_actors.values():

            try:
                viewer.remove_actor(actor, render=False)
            except Exception:
                pass

        self.bond_actors.clear()

    def set_bonds_visible(
        self,
        visible: bool,
    ) -> None:

        for actor in self.bond_actors.values():

            actor.SetVisibility(visible)

    # ==============================================================
    # Polyhedra
    # ==============================================================

    def add_polyhedron_actor(
        self,
        name,
        actor,
    ):

        self.polyhedron_actors[name] = actor

    def remove_polyhedra(
        self,
        viewer,
    ):

        for actor in self.polyhedron_actors.values():

            try:
                viewer.remove_actor(actor, render=False)
            except Exception:
                pass

        self.polyhedron_actors.clear()

    def set_polyhedra_visible(
        self,
        visible: bool,
    ) -> None:

        for actor in self.polyhedron_actors.values():

            actor.SetVisibility(visible)

    # ==============================================================
    # Labels
    # ==============================================================

    def add_label_actor(
        self,
        name,
        actor,
    ):

        self.label_actors[name] = actor

    def remove_labels(
        self,
        viewer,
    ):

        for actor in self.label_actors.values():

            try:
                viewer.remove_actor(actor, render=False)
            except Exception:
                pass

        self.label_actors.clear()

    def set_labels_visible(
        self,
        visible: bool,
    ) -> None:

        for actor in self.label_actors.values():

            actor.SetVisibility(visible)

    # ==============================================================
    # Cell
    # ==============================================================

    def set_cell_actor(
        self,
        actor,
    ):

        self.cell_actor = actor

    def remove_cell(
        self,
        viewer,
    ):

        if self.cell_actor is None:
            return

        try:
            viewer.remove_actor(self.cell_actor, render=False)
        except Exception:
            pass

        self.cell_actor = None

    def set_cell_visible(
        self,
        visible: bool,
    ) -> None:

        if self.cell_actor is not None:

            self.cell_actor.SetVisibility(visible)

    # ==============================================================
    # Getters
    # ==============================================================

    def get_atom_actor(self, name):

        return self.atom_actors.get(name)

    def get_bond_actor(self, name):

        return self.bond_actors.get(name)

    def get_polyhedron_actor(self, name):

        return self.polyhedron_actors.get(name)

    def get_label_actor(self, name):

        return self.label_actors.get(name)

    def get_cell_actor(self):

        return self.cell_actor

    # ==============================================================
    # Visibility
    # ==============================================================

    def update_visibility(
        self,
        settings,
    ) -> None:

        self.set_atoms_visible(True)

        self.set_bonds_visible(
            settings.show_bonds
        )

        self.set_polyhedra_visible(
            settings.show_polyhedra
        )

        self.set_labels_visible(
            settings.show_labels
        )

        self.set_cell_visible(
            settings.show_cell
        )

    # ==============================================================
    # Remove scene
    # ==============================================================

    def remove_scene(
        self,
        viewer,
    ):

        self.remove_atoms(viewer)

        self.remove_bonds(viewer)

        self.remove_polyhedra(viewer)

        self.remove_labels(viewer)

        self.remove_cell(viewer)
