# PyInstaller spec for the Linux build.
#
# Built one-dir rather than one-file. A one-file bundle of this
# dependency set is a ~2 GB self-extracting archive that unpacks to /tmp
# on every launch, which costs tens of seconds each time. The AppImage
# built around this directory is still a single executable file to the
# person using it, and mounts rather than extracts, so it starts fast.

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).parent

datas = [
    # The chevrons and ticks the stylesheet points at. Without these the
    # window still opens, so a missing one is only noticed by eye.
    (str(ROOT / "qvista" / "gui" / "icons"), "qvista/gui/icons"),
    (str(ROOT / "packaging" / "qvista.png"), "."),
]

binaries = []

# Packages that load parts of themselves by name at runtime, which the
# import graph therefore cannot see.
hiddenimports = [
    "vtkmodules",
    "vtkmodules.all",
    "vtkmodules.util",
    "vtkmodules.util.numpy_support",
    "vtkmodules.qt",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "pyvista",
    "pyvistaqt",
    "seekpath",
    "spglib",
    "phonopy",
    "encodings.idna",
]

hiddenimports += collect_submodules("vtkmodules")

# Data files these ship and read at runtime: pymatgen's element tables,
# phonopy's, matplotlib's fonts and styles. A missing table is not an
# import error -- it is a wrong number, much later.
for package in (
    "pymatgen",
    "phonopy",
    "spglib",
    "seekpath",
    "matplotlib",
    "ase",
    "pyvista",
    "monty",
    "uncertainties",
):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass

# Optional: the machine-learning force field. Present in a full build,
# absent in a lean one -- QVista already reports it as missing rather
# than failing, so both are valid.
if os.environ.get("QVISTA_WITH_ML", "1") != "0":
    for package in ("chgnet", "torch"):
        try:
            datas += collect_data_files(package)
            hiddenimports += collect_submodules(package)
        except Exception:
            pass

    try:
        binaries += collect_dynamic_libs("torch")
    except Exception:
        pass

excludes = [
    "tkinter",
    "test",
    "pytest",
    "black",
    "ruff",
    "mypy",
    "Nuitka",
    "IPython",
    "notebook",
    # Qt bindings other than the one in use. Two sets of Qt libraries in
    # one process is a crash, not a warning.
    "PyQt5",
    "PyQt6",
    "PySide2",
]

a = Analysis(
    [str(ROOT / "qvista" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="qvista",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="qvista",
)
