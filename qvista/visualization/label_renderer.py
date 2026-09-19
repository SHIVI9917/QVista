"""
High-performance atom label renderer.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyvista as pv

from qvista.visualization.settings import LabelMode


class LabelRenderer:
    """
    Render atom labels.
    """

    def draw(
        self,
        viewer,
        atoms,
    ) -> None:

        if not atoms:
            return

        # ----------------------------------------------------------
        # Coordinates
        # ----------------------------------------------------------

        points = np.array(
            [
                atom.position
                for atom in atoms
            ],
            dtype=float,
        )

        # ----------------------------------------------------------
        # Labels
        # ----------------------------------------------------------

        labels = []

        counts = defaultdict(int)

        mode = viewer.settings.label_mode

        for atom in atoms:

            # No labels
            if mode == LabelMode.NONE:

                labels.append("")

            # 1, 2, 3, ...
            elif mode == LabelMode.INDEX:

                labels.append(
                    str(atom.original_index + 1)
                )

            # Pb, I, O
            elif mode == LabelMode.SYMBOL:

                labels.append(
                    atom.symbol
                )

            # Pb1, Pb2, I1, I2 (per-element numbering)
            elif mode == LabelMode.SYMBOL_INDEX:

                counts[atom.symbol] += 1

                labels.append(
                    f"{atom.symbol}{counts[atom.symbol]}"
                )

            # Pb1, I2, O3 (global atom numbering)
            elif mode == LabelMode.SYMBOL_GLOBAL:

                labels.append(
                    f"{atom.symbol}{atom.original_index + 1}"
                )

            else:

                labels.append("")

        cloud = pv.PolyData(points)


        actor = viewer.add_point_labels(

            cloud,

            labels,

            font_size=viewer.settings.label_font_size,

            text_color=viewer.settings.label_color,

            shape_opacity=0.0,

            always_visible=True,

            name="labels",

            render=False,

        )

        viewer.cache.add_label_actor(
            "labels",
            actor,
        )


