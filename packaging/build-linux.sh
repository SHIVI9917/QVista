#!/usr/bin/env bash
#
# Build QVista into a single Linux executable: QVista-x86_64.AppImage
#
# Runs without root. Ubuntu's system Python has no ensurepip and the
# machine may be missing Qt's runtime libraries, both of which normally
# mean apt and therefore a password. Instead this fetches a
# self-contained CPython and extracts the handful of libraries Qt needs
# straight out of their .deb files -- which is what has to happen anyway,
# since they have to be bundled for the AppImage to run on a machine that
# does not have them.
#
#   ./build-linux.sh [output-directory]
#
# Environment:
#   QVISTA_WITH_ML=0   leave out CHGNet and torch (much smaller binary)
#   QVISTA_JOBS=N      parallelism for pip

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

OUTPUT="${1:-$HOME/Desktop}"

BUILD="${QVISTA_BUILD_DIR:-$HOME/qvista-build}"
WITH_ML="${QVISTA_WITH_ML:-1}"

PYTHON_VERSION="3.12.11"
PYTHON_RELEASE="20250818"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"

APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"

# Qt 6 dlopens its xcb platform plugin, which links against these. A
# desktop Linux box has most of them; a container or a minimal install
# has almost none, and libxcb-cursor0 is missing even on a full Ubuntu
# 24.04. Bundled rather than required, so the AppImage runs on a bare
# machine -- which is the whole point of shipping one file.
QT_PACKAGES=(
    libxcb-cursor0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-render-util0
    libxcb-util1
    libxcb-xinerama0
    libxcb-xkb1
    libxkbcommon-x11-0
    libxcb-randr0
    libxcb-shape0
    libxcb-sync1
    libxcb-xfixes0
    libxcb-shm0
    libxcb-glx0
)

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mBUILD FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

mkdir -p "$BUILD" "$OUTPUT"

# ------------------------------------------------------------------
# 1. A Python that does not depend on the host
# ------------------------------------------------------------------

PY="$BUILD/python/bin/python3"

if [[ ! -x "$PY" ]]; then
    say "Fetching a self-contained CPython ${PYTHON_VERSION}"

    curl -fL --retry 3 -o "$BUILD/python.tar.gz" "$PYTHON_URL" \
        || die "could not download CPython"

    tar -C "$BUILD" -xzf "$BUILD/python.tar.gz"
    rm -f "$BUILD/python.tar.gz"
fi

[[ -x "$PY" ]] || die "no python at $PY"

"$PY" --version

# ------------------------------------------------------------------
# 2. Dependencies
# ------------------------------------------------------------------

say "Installing dependencies (this is the slow part)"

"$PY" -m pip install --upgrade pip wheel setuptools >/dev/null

# The pinned list, minus the tools that only matter while developing --
# they would otherwise be collected into the binary.
grep -viE '^(black|ruff|mypy|pytest|iniconfig|pluggy|Nuitka|pathspec|pytokens|mypy_extensions)([=<>]|$)' \
    "$ROOT/requirements.txt" > "$BUILD/requirements-runtime.txt"

"$PY" -m pip install -r "$BUILD/requirements-runtime.txt" \
    || die "dependency install failed"

if [[ "$WITH_ML" != "0" ]]; then
    say "Installing CHGNet on CPU torch"

    # The CPU wheel deliberately: the CUDA build is several gigabytes of
    # kernels that a bundled desktop app will not use.
    "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
        || die "torch install failed"

    "$PY" -m pip install chgnet || die "chgnet install failed"
fi

"$PY" -m pip install pyinstaller || die "pyinstaller install failed"

# ------------------------------------------------------------------
# 3. Build
# ------------------------------------------------------------------

# Bundling and packaging are much quicker than the bundle itself, and
# are where the fiddly parts are, so they can be re-run against an
# existing one rather than rebuilding from scratch each time.
if [[ "${QVISTA_REUSE_BUNDLE:-0}" == "1" && -x "$BUILD/dist/qvista/qvista" ]]; then

    say "Reusing the bundle already in $BUILD/dist"

else

    say "Running PyInstaller"

    rm -rf "$BUILD/dist" "$BUILD/work"

    QVISTA_WITH_ML="$WITH_ML" "$PY" -m PyInstaller \
        --noconfirm \
        --clean \
        --distpath "$BUILD/dist" \
        --workpath "$BUILD/work" \
        "$HERE/qvista.spec" \
        || die "PyInstaller failed"

fi

[[ -x "$BUILD/dist/qvista/qvista" ]] || die "no executable was produced"

# ------------------------------------------------------------------
# 3b. Does it actually start?
# ------------------------------------------------------------------
#
# PyInstaller reports success for a bundle that is missing a module it
# never saw imported -- the first build here was missing pyvistaqt, and
# said nothing until the window failed to open. Starting it offscreen
# costs a minute and turns that into a failed build.

say "Checking the bundle starts"

SMOKE="$BUILD/smoke.log"

# On a real display where possible: forcing offscreen only silences Qt,
# and VTK opens its own window regardless, so the run aborts on an X
# error that says nothing about whether the bundle is sound.
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    export QT_QPA_PLATFORM=offscreen
fi

timeout 120 "$BUILD/dist/qvista/qvista" > "$SMOKE" 2>&1 &

SMOKE_PID=$!

# Long enough for every import to have been attempted; an ImportError
# lands within seconds of launch, not a minute in.
sleep 45

if kill -0 "$SMOKE_PID" 2>/dev/null; then
    kill "$SMOKE_PID" 2>/dev/null || true
    wait "$SMOKE_PID" 2>/dev/null || true
    echo "  still running after 45 s -- it started"

else
    wait "$SMOKE_PID" 2>/dev/null || true

    if grep -qE "ModuleNotFoundError|ImportError|Traceback" "$SMOKE"; then
        echo
        sed 's/^/  /' "$SMOKE"
        die "the bundle does not start (see above)"
    fi

    echo "  exited early but without an import error; continuing"
fi

# ------------------------------------------------------------------
# 4. Assemble the AppDir
# ------------------------------------------------------------------

say "Assembling the AppDir"

APPDIR="$BUILD/QVista.AppDir"

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/512x512/apps" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"

cp -a "$BUILD/dist/qvista/." "$APPDIR/usr/bin/"

cp "$HERE/qvista-512.png" \
   "$APPDIR/usr/share/icons/hicolor/512x512/apps/qvista.png"
cp "$HERE/qvista-256.png" \
   "$APPDIR/usr/share/icons/hicolor/256x256/apps/qvista.png"

# appimagetool looks for these two at the top of the AppDir.
cp "$HERE/qvista-256.png" "$APPDIR/qvista.png"
cp "$HERE/qvista.desktop" "$APPDIR/usr/share/applications/qvista.desktop"
cp "$HERE/qvista.desktop" "$APPDIR/qvista.desktop"

# The Qt libraries the host may not have.
say "Bundling Qt's runtime libraries"

# By package name through apt-get download, not by hardcoded URL. Both
# halves of the first attempt were wrong: half the URLs had moved on to
# a newer build number and 404ed, and the ones that did arrive were
# compressed with zstd, which the ar-and-tar extraction could not read.
# apt-get download resolves the current version, dpkg-deb -x reads
# whatever compression it used, and neither needs root.
mkdir -p "$BUILD/debs"

# set +e for the loop: a package that is unavailable is a thing to
# report and carry on from, not a reason to stop. Under set -e the
# no-match `ls` alone was enough to abandon the build silently, which is
# how the previous attempt stopped four packages in with no message.
(
    set +e

    cd "$BUILD/debs" || exit 0

    for package in "${QT_PACKAGES[@]}"; do

        deb=$(ls "${package}"_*.deb 2>/dev/null | head -1) || deb=""

        if [[ -z "$deb" ]]; then

            apt-get download "$package" >/dev/null 2>&1

            deb=$(ls "${package}"_*.deb 2>/dev/null | head -1) || deb=""
        fi

        if [[ -z "$deb" ]]; then
            echo "  unavailable: $package"
            continue
        fi

        rm -rf x

        if ! dpkg-deb -x "$deb" x 2>/dev/null; then
            echo "  could not extract: $package"
            continue
        fi

        # Symlinks as well as the files they point at. A library is
        # loaded by its SONAME -- libxcb-cursor.so.0 -- which in the deb
        # is a symlink to libxcb-cursor.so.0.0.0. Copying only regular
        # files stages the real library under a name nothing asks for,
        # and the loader reports it as missing while it sits right there.
        find x -name '*.so*' \( -type f -o -type l \) \
            -exec cp -a {} "$APPDIR/usr/lib/" \; 2>/dev/null
    done

    rm -rf x
)

# Any SONAME still without a link gets one, read off the library itself
# rather than guessed from the filename.
for lib in "$APPDIR/usr/lib"/*.so.*; do

    [[ -f "$lib" && ! -L "$lib" ]] || continue

    soname=$(objdump -p "$lib" 2>/dev/null | awk '/SONAME/ {print $2}')

    [[ -n "$soname" && ! -e "$APPDIR/usr/lib/$soname" ]] || continue

    ln -sf "$(basename "$lib")" "$APPDIR/usr/lib/$soname"

    echo "  linked $soname -> $(basename "$lib")"
done

echo "  bundled $(ls "$APPDIR/usr/lib" 2>/dev/null | wc -l) libraries"

# ------------------------------------------------------------------
# Does Qt's platform plugin have everything it needs?
# ------------------------------------------------------------------
#
# The plugin is loaded by dlopen, so a missing dependency is not a link
# error at build time -- it is "could not load the Qt platform plugin
# xcb" at launch, with no clue as to which library was absent. Asking
# ldd turns that into a build failure that names the file.

PLUGIN="$APPDIR/usr/bin/PySide6/Qt/plugins/platforms/libqxcb.so"

if [[ -f "$PLUGIN" ]]; then

    MISSING=$(
        LD_LIBRARY_PATH="$APPDIR/usr/lib:$APPDIR/usr/bin" \
            ldd "$PLUGIN" 2>/dev/null | awk '/not found/ {print $1}'
    )

    if [[ -n "$MISSING" ]]; then
        echo
        echo "  the xcb platform plugin is still missing:"
        echo "$MISSING" | sed 's/^/    /'
        die "Qt would fail to start; add the packages providing these"
    fi

    echo "  xcb platform plugin: all dependencies present"
fi

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/bash
# Launcher. Everything QVista needs is inside this file; nothing here
# reaches out to the host except for the graphics driver.
HERE="$(dirname "$(readlink -f "${0}")")"

export LD_LIBRARY_PATH="$HERE/usr/lib:$HERE/usr/bin:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$HERE/usr/bin/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH/platforms"

# Matplotlib and fontconfig both want somewhere writable, and the
# mounted AppImage is read only.
export MPLCONFIGDIR="${XDG_CACHE_HOME:-$HOME/.cache}/qvista/matplotlib"
mkdir -p "$MPLCONFIGDIR"

exec "$HERE/usr/bin/qvista" "$@"
APPRUN

chmod +x "$APPDIR/AppRun"

# ------------------------------------------------------------------
# 5. One file
# ------------------------------------------------------------------

say "Building the AppImage"

TOOL="$BUILD/appimagetool"

if [[ ! -x "$TOOL" ]]; then
    curl -fL --retry 3 -o "$TOOL" "$APPIMAGETOOL_URL" \
        || die "could not download appimagetool"
    chmod +x "$TOOL"
fi

TARGET="$OUTPUT/QVista-x86_64.AppImage"

# --appimage-extract-and-run because FUSE is not always available, and
# needing it would trade one root requirement for another.
ARCH=x86_64 "$TOOL" --appimage-extract-and-run \
    --no-appstream "$APPDIR" "$TARGET" \
    || die "appimagetool failed"

chmod +x "$TARGET"

# ------------------------------------------------------------------
# 6. Does the thing we are about to hand over actually start?
# ------------------------------------------------------------------
#
# The earlier check ran the unpacked bundle, which shares the host's
# libraries and so passed while the AppImage -- which does not -- failed
# on a library that had been staged under the wrong name. Only running
# the finished file tests what is being shipped.

say "Checking the AppImage starts"

FINAL="$BUILD/final.log"

( cd /tmp && timeout 90 "$TARGET" > "$FINAL" 2>&1 ) &

FINAL_PID=$!

sleep 50

if kill -0 "$FINAL_PID" 2>/dev/null; then
    kill "$FINAL_PID" 2>/dev/null || true
    wait "$FINAL_PID" 2>/dev/null || true
    echo "  still running after 50 s -- the window opened"

else
    wait "$FINAL_PID" 2>/dev/null || true

    if grep -qiE "could not load the qt platform plugin|ModuleNotFoundError|ImportError|Traceback" "$FINAL"; then
        echo
        sed 's/^/  /' "$FINAL"
        die "the AppImage does not start (see above)"
    fi

    echo "  exited without an error worth failing over; continuing"
fi

say "Done"

ls -lh "$TARGET"

printf '\n  %s\n\n  Run it with:\n    %s\n\n' \
    "$TARGET" "$TARGET"
