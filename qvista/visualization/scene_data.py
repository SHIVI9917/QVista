"""
Prepared scene data for all renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SceneData:
    """
    All geometry required by the renderers.

    This class contains no rendering code.
    It is simply a container for prepared visualization data.
    """

    # Original structure
    structure: object

    # Visible atoms after periodic replication
    visible_atoms: list = field(default_factory=list)

    # Visible bonds
    graph: object | None = None

    # Visible coordination polyhedra
    polyhedra: list = field(default_factory=list)

    # Label strings (parallel to visible_atoms)
    labels: list[str] = field(default_factory=list)

    # Current boundary
    boundary: object | None = None

    # Viewer settings used to generate this scene
    settings: object | None = None

    # Dirty flags
    geometry_dirty: bool = True
    colors_dirty: bool = True
    visibility_dirty: bool = True
