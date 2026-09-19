"""
Base class for all renderers.
"""

from __future__ import annotations


class BaseRenderer:
    """
    Base renderer class.

    Manages VTK actors owned by a renderer.
    """

    def __init__(self) -> None:

        self._initialized = False

        self._actors = {}

    # ---------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ---------------------------------------------------------

    def initialize(self) -> None:
        """
        Allocate renderer resources.
        """
        self._initialized = True

    # ---------------------------------------------------------

    def clear(
        self,
        viewer,
    ) -> None:
        """
        Remove all actors.
        """

        for actor in self._actors.values():

            try:
                viewer.remove_actor(actor)
            except Exception:
                pass

        self._actors.clear()

        self._initialized = False

    # ---------------------------------------------------------

    def show(self) -> None:

        for actor in self._actors.values():

            actor.SetVisibility(True)

    # ---------------------------------------------------------

    def hide(self) -> None:

        for actor in self._actors.values():

            actor.SetVisibility(False)
