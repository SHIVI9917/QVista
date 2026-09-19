"""
Screenshot every dialog the tutorial describes.

Two things matter here, and the previous version of this script got the
first one wrong.

**The application is built through :func:`create_application`**, exactly
as ``python -m qvista`` does. That is what applies the theme stylesheet.
Constructing a bare ``QApplication`` instead gives Qt's native widgets,
which is not what anybody running QVista sees -- and on the INCAR and
KPOINTS dialogs it renders the field labels white on white, so the
screenshots were not merely dated but unreadable.

**Fonts are registered by path.** Qt's offscreen platform starts with an
empty font database on Windows, so every label would otherwise come out
as tofu boxes. Offscreen is worth that small ceremony: it needs no
display, and it cannot throw thirty windows over whatever else is on the
screen.

The 3D viewport is not captured here -- VTK cannot get an OpenGL context
under the offscreen platform. The main window is handled by
``capture_window.py``, which composites the viewport in.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from qvista.app import create_application  # noqa: E402

app = create_application()

from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402

_families: list[str] = []

for _name in (
    "segoeui.ttf",
    "seguisb.ttf",
    "seguivar.ttf",
    "arial.ttf",
    "calibri.ttf",
    "tahoma.ttf",
    "consola.ttf",
    "cascadiacode.ttf",
    "cour.ttf",
):
    _path = Path(r"C:\Windows\Fonts") / _name

    if _path.is_file():
        _families += QFontDatabase.applicationFontFamilies(
            QFontDatabase.addApplicationFont(str(_path))
        )

if "Segoe UI" in _families:
    app.setFont(QFont("Segoe UI", 10))

print(f"fonts: {len(set(_families))} families", flush=True)

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

DEMO = Path(sys.argv[1] if len(sys.argv) > 1
            else "LiMnO2-1819592.vasp").resolve()

done: list[tuple[str, int]] = []
failed: list[tuple[str, str]] = []


def shot(name, build, w=1040, h=780):
    """Build a dialog, grab it, and say what happened."""

    try:
        widget = build()

    except Exception as exc:
        failed.append((name, f"{type(exc).__name__}: {exc}"))
        return

    try:
        widget.resize(w, h)
        widget.show()

        for _ in range(30):
            app.processEvents()

        path = OUT / f"{name}.png"

        if widget.grab().save(str(path)) and path.stat().st_size > 4000:
            done.append((name, path.stat().st_size))
            print(
                f"  {name:26s} {path.stat().st_size // 1024:5d} KB",
                flush=True,
            )
        else:
            failed.append((name, "grab produced an empty image"))

        widget.close()

    except Exception as exc:
        failed.append((name, f"{type(exc).__name__}: {exc}"))


print("capturing:", flush=True)

# ------------------------------------------------------------------
# VASP input
# ------------------------------------------------------------------

from qvista.gui.dialogs.incar_dialog import IncarDialog  # noqa: E402

for job, label in (
    ("relax", "incar-optimization"),
    ("static", "incar-single-point"),
    ("md", "incar-md"),
    ("phonon", "incar-phonon"),
):
    shot(
        label,
        lambda j=job: IncarDialog(
            elements=["Li", "Mn", "O"], counts=[4, 4, 8], job=j
        ),
        1120,
        860,
    )

from qvista.gui.dialogs.potcar_dialog import PotcarDialog  # noqa: E402

shot(
    "potcar",
    lambda: PotcarDialog(
        elements=["Li", "Mn", "O"],
        family="PBE",
        counts={"Li": 4, "Mn": 4, "O": 8},
    ),
    980,
    700,
)

from qvista.gui.dialogs.kpoints_dialog import (  # noqa: E402
    CustomisedKpointsDialog,
    KSpacingDialog,
)
from pymatgen.core import Structure as PmgStructure  # noqa: E402

pmg = PmgStructure.from_file(str(DEMO))

shot("kpoints-kspacing-gamma",
     lambda: KSpacingDialog(pmg, "Gamma"), 900, 620)
shot("kpoints-customised-mp",
     lambda: CustomisedKpointsDialog(pmg, "Monkhorst-Pack"), 900, 620)

# ------------------------------------------------------------------
# Structure editing and optimisation
# ------------------------------------------------------------------

from qvista.gui.dialogs.relax_dialog import RelaxDialog  # noqa: E402

shot("optimization-chgnet", lambda: RelaxDialog(natoms=16), 760, 620)

from qvista.gui.dialogs.selective_dialog import (  # noqa: E402
    SelectiveDynamicsDialog,
)

shot("selective-dynamics",
     lambda: SelectiveDynamicsDialog(start=str(DEMO)), 1000, 760)

from qvista.gui.dialogs.convert_dialog import ConvertDialog  # noqa: E402

shot("convert", lambda: ConvertDialog(start=str(DEMO)), 900, 640)

# ------------------------------------------------------------------
# Electronic structure
# ------------------------------------------------------------------

from qvista.gui.dialogs.electronic_dialog import (  # noqa: E402
    BandCentreDialog,
    BandDialog,
    DosDialog,
)

shot("band-structure", lambda: BandDialog(), 1060, 800)
shot("dos", lambda: DosDialog(), 1060, 800)
shot("band-centre", lambda: BandCentreDialog(), 1120, 840)

from qvista.gui.dialogs.orbital_dialog import OrbitalDialog  # noqa: E402

shot("orbital-projected", lambda: OrbitalDialog(view="spd"), 1000, 760)
shot("orbital-resolved", lambda: OrbitalDialog(view="resolved"), 1000, 760)

# ------------------------------------------------------------------
# Bonding
# ------------------------------------------------------------------

from qvista.gui.dialogs.bader_dialog import BaderDialog  # noqa: E402

shot("bader", lambda: BaderDialog(), 1000, 740)

from qvista.gui.dialogs.ddec_dialog import DdecDialog  # noqa: E402

shot("ddec6", lambda: DdecDialog(), 1000, 740)

from qvista.gui.dialogs.cohp_dialog import CohpDialog  # noqa: E402

shot("cohp", lambda: CohpDialog(), 1000, 740)

# ------------------------------------------------------------------
# Phonons
# ------------------------------------------------------------------

from qvista.gui.dialogs.phonon_vasp_dialog import (  # noqa: E402
    DimensionsDialog,
    VaspPostprocessDialog,
)

shot("phonon-vasp-dimensions", lambda: DimensionsDialog(), 700, 210)
shot("phonon-vasp-postprocess", lambda: VaspPostprocessDialog(), 1040, 780)

# ------------------------------------------------------------------
# NEB
# ------------------------------------------------------------------

from qvista.gui.dialogs.neb_vasp_dialog import (  # noqa: E402
    NebPostprocessDialog,
    NebPreprocessDialog,
)

shot("neb-vasp-preprocess", lambda: NebPreprocessDialog(), 1040, 800)
shot("neb-vasp-postprocess", lambda: NebPostprocessDialog(), 1060, 820)

from qvista.gui.dialogs.neb_chgnet_dialog import NebChgnetDialog  # noqa: E402

shot("neb-chgnet", lambda: NebChgnetDialog(), 1060, 840)

# ------------------------------------------------------------------
# Molecular dynamics and transport
# ------------------------------------------------------------------

from qvista.gui.dialogs.dynamics_dialog import MDDialog  # noqa: E402

shot("md-run-dynamics", lambda: MDDialog(natoms=16), 820, 700)

from qvista.gui.dialogs.md_vasp_dialog import MdVaspDialog  # noqa: E402

shot("md-vasp", lambda: MdVaspDialog(), 1040, 780)

from qvista.gui.dialogs.msd_dialog import MsdDialog  # noqa: E402

shot("msd-diffusion", lambda: MsdDialog(), 1040, 780)

from qvista.gui.dialogs.correlations_dialog import (  # noqa: E402
    CorrelationsDialog,
)

for mode, label in (
    ("rdf", "rdf"),
    ("vanhove", "van-hove"),
    ("residence", "residence-time"),
):
    shot(label, lambda m=mode: CorrelationsDialog(mode=m), 1040, 780)

from qvista.gui.dialogs.kinetics_dialog import KineticsDialog  # noqa: E402

shot("diffusion-kinetics", lambda: KineticsDialog(), 1000, 760)

# The electronic-conductivity dialog is deliberately absent: the
# Conductivity section was removed from QVista, so there is nothing left
# to photograph.

# ------------------------------------------------------------------
# Mechanical
# ------------------------------------------------------------------

from qvista.gui.dialogs.elastic_vasp_dialog import (  # noqa: E402
    ElasticVaspDialog,
)

shot("mechanical-vasp", lambda: ElasticVaspDialog(), 1040, 780)

# ------------------------------------------------------------------
# Plot design
# ------------------------------------------------------------------

from qvista.gui.dialogs.style_window import StyleWindow  # noqa: E402

shot("plot-design", lambda: StyleWindow(), 900, 760)

# ------------------------------------------------------------------

print(f"\ncaptured {len(done)}", flush=True)

if failed:
    print(f"\nfailed {len(failed)}:", flush=True)
    for name, why in failed:
        print(f"  {name:26s} {why[:90]}", flush=True)
