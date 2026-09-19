# Building QVista as a Linux binary

The result is a single executable file, `QVista-x86_64.AppImage`, that
carries Python, Qt, VTK, the whole scientific stack and (optionally)
CHGNet with PyTorch inside it. Nothing needs installing on the machine
that runs it: make it executable and double-click.

```bash
./build-linux.sh                 # writes to ~/Desktop
./build-linux.sh /some/where     # or wherever you say
```

## What you cannot do

**Cross-compiling is not possible.** Neither PyInstaller nor Nuitka can
produce a Linux binary from Windows: both work by bundling the
interpreter and the compiled extension modules that are already on the
build machine, and those are Windows `.pyd` files. A Linux binary has to
be built on Linux.

On Windows that means WSL, which is what this was built with:

```powershell
wsl -d Ubuntu
cd /mnt/c/Users/Asus/Desktop/BTP/QVista/QVista/packaging
./build-linux.sh /mnt/c/Users/Asus/Desktop
```

## Why it does not need root

Ubuntu 24.04's system Python has no `ensurepip`, and the machine is
missing some of the libraries Qt 6 loads at runtime. Both would normally
mean `apt` and a password. Instead the script:

- downloads a self-contained CPython from `python-build-standalone`,
  which is also a better base for a portable binary than a distribution
  Python that expects the distribution around it;
- extracts the missing Qt libraries out of their `.deb` files with `ar`,
  which is what has to happen anyway — they must be *bundled* for the
  AppImage to run on a machine that does not have them, and requiring
  them to be installed first would defeat the point.

## Why one directory inside one file

PyInstaller builds one-dir here, not one-file. A one-file bundle of this
dependency set is a ~2 GB self-extracting archive that unpacks to `/tmp`
on every launch, which costs tens of seconds each time. The AppImage
wrapped around the directory is still a single file to the person using
it, and it *mounts* rather than extracts, so startup is normal.

## Size

| Build | Approximate size |
|---|---|
| `QVISTA_WITH_ML=0` — no force field | 700 MB – 1 GB |
| Default — CHGNet on CPU PyTorch | 2 – 2.5 GB |

Torch is installed from PyTorch's CPU index deliberately. The default
CUDA wheel is several gigabytes of GPU kernels that a bundled desktop
application will not use.

Without the force field, QVista still runs everything else — it reports
the force field as missing rather than failing, which is why the lean
build is a legitimate option rather than a broken one.

## glibc: the one real portability limit

An AppImage does not bundle glibc, so **it runs on the distribution it
was built on and anything newer, not older**. Built on Ubuntu 24.04
(glibc 2.39) it will not start on Ubuntu 22.04 or RHEL 8.

If it has to run on older machines, build on the oldest one you intend
to support. There is an Ubuntu 22.04 distribution already installed here:

```powershell
wsl -d Ubuntu-22.04
```

The script is unchanged; only the base matters. Building on 22.04
(glibc 2.35) covers 22.04, 24.04 and current Fedora.

## What is in here

| File | |
|---|---|
| `build-linux.sh` | the whole pipeline |
| `qvista.spec` | PyInstaller spec — hidden imports and data files |
| `qvista.desktop` | the menu entry, file associations and icon name |
| `qvista-512.png` etc. | the icon, at the sizes a desktop asks for |

### The parts of the spec that matter

Three groups of things the import graph cannot see on its own, and each
fails differently:

- **`vtkmodules`** is imported by name at runtime, so without the
  explicit `collect_submodules` the 3D viewer is missing entirely.
- **Data files** — `pymatgen`'s element tables, `phonopy`'s, matplotlib's
  fonts. A missing table is not an import error; it is a wrong number,
  much later, which is the worst way for this to go wrong.
- **Other Qt bindings are excluded.** Two sets of Qt libraries loaded
  into one process is a crash, not a warning.

## The build checks itself twice

Both gates exist because something got past the absence of one.

1. **After PyInstaller**, the unpacked bundle is launched for 45 seconds.
   PyInstaller reports success for a bundle missing a module it never
   saw imported — the first build here was missing `pyvistaqt` and said
   nothing until the window failed to open.
2. **After the AppImage is built**, the finished file is launched for 50
   seconds. The first check is not enough on its own: the unpacked
   bundle borrows the host's libraries, so it passed while the AppImage,
   which does not, failed on `libxcb-cursor.so.0`.

Only the second one tests what is actually being shipped.

There is also an `ldd` check on Qt's xcb plugin, which is worth keeping
but does not catch everything: Qt **dlopens** `libxcb-cursor.so.0` by
name rather than linking it, so `ldd` reports nothing missing while the
plugin still refuses to load.

### The SONAME trap

Libraries are loaded by their SONAME — `libxcb-cursor.so.0` — which in
a `.deb` is a symlink to `libxcb-cursor.so.0.0.0`. Staging only regular
files puts the real library in place under a name nothing asks for, and
the loader reports it missing while it sits right there. The script
copies the symlinks too, and regenerates any SONAME link that is absent
by reading it out of the library with `objdump`.

## If it does not start

Run it from a terminal — the window is silent, the terminal is not:

```bash
./QVista-x86_64.AppImage
QT_DEBUG_PLUGINS=1 ./QVista-x86_64.AppImage    # if Qt cannot find a platform plugin
```

To look inside:

```bash
./QVista-x86_64.AppImage --appimage-extract
```

On a headless or remote machine, Qt needs a display. `xvfb-run
./QVista-x86_64.AppImage` gets it far enough to prove it starts.
