# QVista

Integrated computational materials properties analysis platform.

A PySide6 desktop application for inspecting and analysing crystal
structures: 3D visualisation, symmetry analysis, phonons, band structures and
mechanical properties.

## Input

QVista reads a structure from a local file. Supported formats are detected
automatically -- no need to say which one you have:

| Format | Notes |
|---|---|
| POSCAR / CONTCAR | Recognised by filename, with or without an extension |
| CIF | |
| Extended XYZ | Lattice read from the comment line (ASE style) |
| Plain XYZ | No lattice, so it is treated as a molecule |

A plain XYZ carries no cell. Volume, space group and crystal system are
undefined for it and display as `N/A`, and analyses that need a lattice
(k-paths, phonons, elastic tensors) are refused with an explanation.

There is no Materials Project integration and no API key -- input is always a
file you supply.

## Installation

QVista is a Python application, not a packaged binary. You clone this
repository and install it into a virtual environment of its own. The steps are
the same shape on both platforms -- system prerequisites, clone, virtual
environment, install -- and differ only in the first step.

### What you need

| | |
|---|---|
| **Python** | **3.11 or newer.** A hard floor, not a preference: the code uses `StrEnum`, which arrived in 3.11. On 3.10 QVista fails at import. |
| **Git** | to clone the repository |
| **Disk** | roughly 1.5 GB for the environment -- PySide6, VTK and pymatgen are large |
| **Graphics** | OpenGL 3.2 or newer for the 3D viewport. Any desktop GPU works, and so does software rendering through Mesa. |

**VASP is not required and is never launched.** QVista writes VASP inputs and
reads VASP outputs. You can prepare runs on a laptop, run them on a cluster,
and read the results back on the laptop. The same is true of POTCAR files:
QVista needs a path to your own licensed copy, which you set inside the
application under **File > Set POTCAR Library**. None are distributed here.

---

### Windows

**1. Install Python 3.11 or newer.**

Download it from [python.org](https://www.python.org/downloads/windows/). On
the installer's first screen, tick **"Add python.exe to PATH"** -- without it,
the commands below will not be found.

Then open a new PowerShell window and confirm:

```powershell
python --version
```

It must report 3.11 or higher. If it reports 3.10 or lower, or if the command
opens the Microsoft Store instead, the PATH entry did not take; reinstall with
the box ticked.

**2. Clone the repository.**

```powershell
git clone https://github.com/CMCLAB-IITG/QVista.git
cd QVista
```

**3. Create and activate a virtual environment.**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses with a script-execution error, allow it for this window
only and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

On `cmd.exe` the activation command is `.venv\Scripts\activate.bat` instead.

Your prompt should now begin with `(.venv)`.

**4. Install QVista.**

```powershell
python -m pip install --upgrade pip
pip install -e .
```

**5. Run it.**

```powershell
qvista
```

You can also double-click `QVista.bat` in the project folder, which activates
the environment and launches the application in one step.

---

### Linux

**1. Install the system libraries.**

This command covers two unrelated things, and the first is the reason it
cannot be deferred to whenever you next care about graphics.

`python3-venv` is in the list, and it is what step 4 uses. Debian and Ubuntu
package `venv` separately from Python itself, so on a stock Ubuntu 24.04 it is
simply absent -- and `python3 -m venv .venv` then fails at once with `ensurepip
is not available`, long before anything has to be drawn on a screen.

The rest are Qt's. It does not bundle the dependencies of its X11 platform
plugin, and a minimal install has almost none of them; skipping those produces
the confusing `xcb` error described under *Troubleshooting*.

So this is the step that is easy to skip and should not be. Skipping it does
not merely cost you the window -- it stops the install at step 4.

On Debian and Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip \
    libgl1 libegl1 libglib2.0-0 libdbus-1-3 libfontconfig1 \
    libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
    libxcb-shm0 libxcb-sync1 libxcb-util1 libxcb-xfixes0 \
    libxcb-xinerama0 libxcb-xkb1 libxcb-glx0
```

On Fedora and RHEL the equivalents are `mesa-libGL`, `mesa-libEGL`,
`libxkbcommon-x11`, `fontconfig`, `dbus-libs` and the `xcb-util-*` family.
Those package names have not been tested against this list.

**2. Check your Python version.**

```bash
python3 --version
```

QVista needs **3.11 or newer**. Ubuntu 24.04 and later ship 3.12 and need
nothing further. **Ubuntu 22.04 ships Python 3.10, which is too old** -- see
*Troubleshooting* before continuing.

**3. Clone the repository.**

```bash
git clone https://github.com/CMCLAB-IITG/QVista.git
cd QVista
```

**4. Create and activate a virtual environment.**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**5. Install QVista.**

```bash
python -m pip install --upgrade pip
pip install -e .
```

**6. Run it.**

```bash
qvista
```

---

### Confirm the installation

Install the development extra and run the test suite:

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

Around 1740 tests should pass with no failures. The suite exercises the
analysis code without opening a window, so it is also the quickest way to tell
a broken installation from a broken display.

For the graphical side, launch `qvista` and open a structure -- the repository
ships `LiMnO2-1819592.vasp` at its root for exactly this.

### Installing the exact pinned versions

`pip install -e .` resolves current releases that satisfy QVista's lower
bounds, which is what you want for ordinary use. `requirements.txt` instead
records the exact versions this was last developed and tested against. Use it
when you need to reproduce a result precisely, or when a fresh resolve has
pulled in an incompatible release:

```bash
pip install -r requirements.txt
```

### Optional: the machine-learning force field

Phonons and mechanical properties come two ways.

**CHGNet** evaluates the forces on your own machine and answers in seconds. It
is fast screening, not DFT, and it needs one of the potentials below.

**VASP** is the first-principles route and needs none of them. Preprocessing
asks for the supercell dimensions, then for a method -- `Config_1`,
`Config_2`, ... for finite displacement, or a single `DFPT` folder -- and
*Generate VASP Inputs* fills those folders with INCAR, KPOINTS and POTCAR.
Postprocessing reads the `vasprun.xml` files back afterwards to build the
force constants, the dispersion and the phonon DOS.

The force field is deliberately *not* a required dependency, because all three
options pull in torch, a multi-gigabyte download, and everything else works
without it:

```bash
pip install -e ".[ml]"
```

Three potentials are supported, and QVista uses whichever is installed, naming
it in the console and on exported figures so a result is never attributed to
the wrong model:

| Potential | Install | Notes |
|---|---|---|
| **CHGNet** | `pip install -e ".[ml]"` | Recommended. Ships wheels, installs anywhere. |
| **MACE-MP** | `pip install -e ".[mace]"` | Also ships wheels. |
| **MatterSim** | `pip install -e ".[mattersim]"` | **No wheels** -- builds from source, so on Windows it also needs the [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). |

If you keep model weights outside the environment, point QVista at them with
**File > Set CHGNet Path**.

Everything else -- visualisation, structure properties, symmetry analysis, the
high-symmetry k-path and the whole VASP phonon route -- works without any of
them. The force-field workflows detect that none is installed and say so, with
install instructions, rather than failing.

These are surrogate models, not DFT. Treat their results as fast screening,
and use the VASP route when the number has to be defensible. Every figure is
stamped with which of the two produced it.

---

### Troubleshooting

**`qvista: command not found` after installing.**
The virtual environment is not active. Re-run `source .venv/bin/activate`
(Linux) or `.venv\Scripts\Activate.ps1` (Windows). You can always fall back to
`python -m qvista` from the project root.

**`ModuleNotFoundError: No module named 'PySide6'` -- or `ase`, or `pyvista`.**
You are running a different Python from the one you installed into. Check with
`python -c "import sys; print(sys.executable)"`; the path should be inside
`.venv`.

**`TypeError: dataclass() got an unexpected keyword argument 'slots'`.**
Python 3.9 or older. See the version requirement above.

**`ImportError: cannot import name 'StrEnum' from 'enum'`.**
Python 3.10. Same cause -- QVista needs 3.11 or newer.

**Ubuntu 22.04: Python is 3.10, and `libxcb-cursor0` will not install.**
Both are real limits of that release. `libxcb-cursor0` first appears in Ubuntu
23.04, and Qt 6.5 and later require it. The cleanest fix is a newer Python
alongside the system one:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
```

then fetch the one missing library from a later release:

```bash
wget http://archive.ubuntu.com/ubuntu/pool/universe/x/xcb-util-cursor/libxcb-cursor0_0.1.4-1_amd64.deb
sudo dpkg -i libxcb-cursor0_0.1.4-1_amd64.deb
```

Upgrading to Ubuntu 24.04 avoids both problems and is the better answer where
that is open to you.

**`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`.**
A system library from step 1 is missing. Ask Qt which one:

```bash
QT_DEBUG_PLUGINS=1 qvista 2>&1 | grep -i "cannot open shared object"
```

The line names the exact `.so`, and the package that provides it is in the
`apt install` list above.

**The window opens but the 3D viewport is black.**
VTK cannot get an OpenGL context. Over SSH, forward it with `ssh -X` and
install `libgl1` on the remote machine. Inside a VM, enable 3D acceleration.
Failing both, force software rendering:

```bash
LIBGL_ALWAYS_SOFTWARE=1 qvista
```

**Running under WSL.** WSLg on Windows 11 runs Qt applications with no extra
configuration -- follow the Linux instructions inside the WSL shell. On
Windows 10 you would need a separate X server and a `DISPLAY` export;
installing natively on Windows is simpler.

**`pip install -e .` fails while building a dependency.** Almost always
`mattersim`, which publishes no wheels. It is optional -- install the base
package first and add a force field afterwards.

## Running

Once installed, any of these start the application:

```bash
qvista                 # the installed command, from anywhere
python -m qvista       # equivalent, from the project root
python -m qvista.main  # the same entry point, named explicitly
```

On Windows, `QVista.bat` does the same by double-click.

QVista opens with an empty workspace. Use **File > Open Project** to load a structure.

## Tutorial

[`docs/tutorial/`](docs/tutorial/README.md) is a complete walkthrough, one
page per part of the program, in the order you would actually use them:
opening a structure, POTCAR, INCAR, KPOINTS, optimisation, symmetry,
electronic structure, bonding and charge, phonons, NEB, molecular dynamics,
mechanical properties, ionic conductivity, plots and export, and structure
editing.

Each page gives three things: the chemistry the tool is computing, the
procedure as pseudocode so it is clear what actually happens, and
screenshots of the running application at each step. The screenshots are
captured programmatically by the scripts in
[`docs/tutorial/tools/`](docs/tutorial/tools/README.md), so they can be
regenerated when the interface changes rather than drifting away from it.
The plots are from real calculations -- a real `DOSCAR`, a real 14 ps VASP
AIMD run at 1600 K -- not mock-ups.

## Layout

```
qvista/
  analyses/       analysis workflows and their shared scaffolding
  controllers/    coordinate models and services
  core/           configuration and logging
  gui/            main window, menus, docks, widgets
  models/         dataclasses describing project state
  services/       file parsing and structure analysis
  visualization/  VTK/PyVista rendering pipeline

docs/
  tutorial/       the walkthrough above, with its screenshots and capture tools
tests/            the test suite
```

## Citation

If you use QVista in your research, please cite:

```bibtex
@article{paitandi_mgcob_cathode,
  author  = {Somnath Paitandi and Shivansh Upadhyay and Kalishankar Bhattacharyya},
  title   = {Computational Investigation of Ternary Mg-Co-B Compounds
             as Cathode Materials for Mg-Ion Batteries
  journal = {Physical Chemistry Chemical Physics},
  year    = {2026},
  month   = {},
  volume  = {},
  number  = {},
  pages   = {},
  issn    = {},
  doi     = {DOI: 10.1039/D6CP01933A},
}
```
