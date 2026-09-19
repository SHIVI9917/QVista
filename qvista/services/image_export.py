"""
Saving the viewport as an image.

A structure file records where the atoms are; a figure records what was
being looked at. The second is not recoverable from the first -- the
orientation, the boundary, the bonds that were switched off, the
representation -- so it has to be exported while it is on screen.

Two things here are less obvious than they look.

Resolution is a supersampling factor rather than a pixel size. VTK can
be asked for an exact output size, but only by resizing the render
window and putting it back, and that window is a live Qt widget in a
layout. Rendering at a multiple of the on-screen size touches nothing
and gives the same result, so the dialog offers a scale and says what it
comes to in pixels.

Transparency is not available in every format. JPEG and BMP have no
alpha channel at all, and asking for one does not produce an opaque
image -- it raises, several layers down, after the render has already
happened. :func:`save_image` refuses the combination up front rather
than letting that surface as a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Supersampling bounds. Above about 8x an ordinary viewport is into
#: tens of megapixels and the render starts to fail on the GPU's texture
#: limits rather than producing anything more useful.
MIN_SCALE = 1
MAX_SCALE = 8

#: What the dialog opens on: enough for a figure at print size without
#: making the first export feel slow.
DEFAULT_SCALE = 2


@dataclass(frozen=True)
class ImageFormat:
    """One way of writing the viewport out."""

    #: What the dropdown shows.
    name: str

    #: The extension written, including the dot.
    suffix: str

    #: True when the format stores geometry rather than pixels. Those
    #: ignore the scale entirely -- there is no resolution to choose.
    vector: bool = False

    #: True when the format has an alpha channel.
    alpha: bool = False

    #: Shown beside the format so the choice is not made on extension
    #: alone.
    note: str = ""

    @property
    def filter(self) -> str:
        """A Qt file-dialog filter for this format alone."""

        return f"{self.name} image (*{self.suffix})"


#: Raster first, since that is what nearly every use wants, and PNG
#: first of those: lossless, alpha, and understood everywhere.
FORMATS: tuple[ImageFormat, ...] = (
    ImageFormat(
        "PNG",
        ".png",
        alpha=True,
        note="Lossless, supports transparency. The default for figures.",
    ),
    ImageFormat(
        "JPEG",
        ".jpg",
        note="Lossy. Smaller files, but compression artefacts around "
        "atom edges and no transparency.",
    ),
    ImageFormat(
        "TIFF",
        ".tif",
        alpha=True,
        note="Lossless and uncompressed. Large files; asked for by some "
        "journals.",
    ),
    ImageFormat(
        "BMP",
        ".bmp",
        note="Uncompressed, no transparency. Rarely the right choice.",
    ),
    ImageFormat(
        "SVG",
        ".svg",
        vector=True,
        note="Vector. Scales without pixelating, but a 3D scene exports "
        "as many overlapping shapes and the file can be large.",
    ),
    ImageFormat(
        "PDF",
        ".pdf",
        vector=True,
        note="Vector, for dropping straight into a document.",
    ),
    ImageFormat(
        "EPS",
        ".eps",
        vector=True,
        note="Vector, for LaTeX workflows that still want it.",
    ),
)

FORMATS_BY_SUFFIX: dict[str, ImageFormat] = {
    fmt.suffix: fmt for fmt in FORMATS
}

#: Extensions that mean the same format as one of the above.
ALIASES: dict[str, str] = {
    ".jpeg": ".jpg",
    ".tiff": ".tif",
    ".ps": ".eps",
}


def format_for(suffix: str) -> ImageFormat | None:
    """
    The format a file extension names, or ``None``.

    Case-insensitive, and ``.jpeg`` and ``.tiff`` land on the same
    formats as their shorter spellings -- a user who types either has
    said what they want clearly enough.
    """

    lowered = suffix.lower()

    if not lowered.startswith("."):
        lowered = f".{lowered}"

    lowered = ALIASES.get(lowered, lowered)

    return FORMATS_BY_SUFFIX.get(lowered)


def output_size(
    window: tuple[int, int], scale: int
) -> tuple[int, int]:
    """
    The pixel size an export will come to.

    Shown live beside the scale, because "4x" means nothing without
    knowing what it is four times.
    """

    width, height = window

    factor = clamp_scale(scale)

    return (int(width) * factor, int(height) * factor)


def clamp_scale(scale: int) -> int:
    """Hold the supersampling factor inside what will actually render."""

    try:
        value = int(scale)

    except (TypeError, ValueError):
        return DEFAULT_SCALE

    return max(MIN_SCALE, min(MAX_SCALE, value))


def supports_transparency(fmt: ImageFormat) -> bool:
    """
    Whether a transparent background can be asked for.

    Only the raster formats with an alpha channel. A vector export
    carries no background at all unless one is drawn, so the question
    does not arise there either.
    """

    return fmt.alpha and not fmt.vector


def with_suffix(path, fmt: ImageFormat) -> Path:
    """
    ``path`` with the format's extension, unless it already names it.

    A user who typed ``figure.tiff`` gets that filename rather than
    ``figure.tiff.tif``, since the alias resolves to the same format.
    """

    path = Path(path)

    existing = format_for(path.suffix)

    if existing is fmt and path.suffix:
        return path

    return path.with_suffix(fmt.suffix)


def _set_marker_visible(plotter, visible: bool) -> bool:
    """
    Show or hide the orientation marker, reporting whether it was on.

    Returned so the caller can put it back exactly as it was: hiding it
    for an export must not leave the viewport changed afterwards.

    The marker is a widget rather than an actor, and which attribute
    holds it has moved between VTK versions, so this asks the renderer
    and gives up quietly rather than guessing.
    """

    # PyVista calls it ``axes_widget``; older versions used
    # ``marker_widget``. Asking for both beats pinning a version.
    widget = None

    for name in ("axes_widget", "marker_widget"):

        widget = getattr(plotter.renderer, name, None)

        if widget is not None:
            break

    if widget is None:
        return False

    try:
        was_on = bool(widget.GetEnabled())

        widget.SetEnabled(1 if visible else 0)

        return was_on

    except Exception:
        return False


def save_image(
    plotter,
    path,
    fmt: ImageFormat | None = None,
    scale: int = DEFAULT_SCALE,
    transparent: bool = False,
    axes: bool = True,
) -> Path:
    """
    Write what the plotter is showing to ``path``.

    Returns the path actually written, which may differ from the one
    given by its extension.

    ``transparent`` is dropped rather than honoured for a format with no
    alpha channel. Passing it through would raise from inside the image
    writer, after the scene had already been rendered at full scale --
    a slow failure with a traceback that says nothing about the checkbox
    that caused it.
    """

    path = Path(path)

    if fmt is None:
        fmt = format_for(path.suffix) or FORMATS[0]

    path = with_suffix(path, fmt)

    path.parent.mkdir(parents=True, exist_ok=True)

    # The orientation marker's labels are 2D captions at a fixed point
    # size while its arrows are 3D geometry. Enlarging the render pulls
    # the two apart, and the letters end up oversized and running into
    # each other. Being able to leave it out is the reliable way to get
    # a clean figure; see the module docstring.
    restore = None

    if not axes:
        restore = _set_marker_visible(plotter, False)

    try:

        if fmt.vector:

            # Resolution-independent, so the scale does not apply. VTK's
            # exporter takes the scene as it stands.
            plotter.save_graphic(str(path))

            return path

        plotter.screenshot(
            str(path),
            scale=clamp_scale(scale),
            transparent_background=bool(transparent)
            and supports_transparency(fmt),
            return_img=False,
        )

        return path

    finally:

        # Whatever happened, the viewport goes back as it was -- an
        # export that fails must not also cost the user their axes.
        if restore:
            _set_marker_visible(plotter, True)


def dialog_filter() -> str:
    """
    The whole filter string for a save dialog, every format at once.

    The combined entry goes first so the dialog opens on it and the
    extension typed decides the format, which is how most people expect
    a save box to behave.
    """

    every = " ".join(f"*{fmt.suffix}" for fmt in FORMATS)

    parts = [f"Image ({every})"]

    parts += [fmt.filter for fmt in FORMATS]

    return ";;".join(parts)
