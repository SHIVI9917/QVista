"""
Exporting the viewport as an image.

The failure worth guarding is the one that costs a full render before it
raises: asking a format with no alpha channel for a transparent
background. It does not produce an opaque image, it raises from inside
the image writer with a message about colour modes, several layers below
the checkbox that caused it.
"""

from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista")

from qvista.services import image_export as ie  # noqa: E402

WINDOW = (200, 150)


@pytest.fixture
def plotter():
    """An off-screen view of something with a silhouette."""

    try:
        view = pv.Plotter(off_screen=True, window_size=WINDOW)

    except Exception as exc:  # pragma: no cover - no GL available
        pytest.skip(f"no render window: {exc}")

    view.add_mesh(pv.Sphere(radius=1.0), color="tan")
    view.render()

    yield view

    view.close()


def opened(path):
    """The written image, via Pillow."""

    Image = pytest.importorskip("PIL.Image")

    return Image.open(path)


# ----------------------------------------------------------------
# The format table
# ----------------------------------------------------------------


def test_the_default_format_is_png():
    """Lossless, alpha, and understood everywhere."""

    assert ie.FORMATS[0].name == "PNG"


def test_every_format_has_a_distinct_suffix():

    suffixes = [fmt.suffix for fmt in ie.FORMATS]

    assert len(suffixes) == len(set(suffixes))


def test_every_format_says_what_it_is_for():

    for fmt in ie.FORMATS:
        assert fmt.note, fmt.name


def test_a_suffix_finds_its_format():

    assert ie.format_for(".png").name == "PNG"
    assert ie.format_for("png").name == "PNG"
    assert ie.format_for(".PNG").name == "PNG"


def test_the_longer_spellings_are_the_same_format():
    """Someone who types ``.jpeg`` or ``.tiff`` has been clear enough."""

    assert ie.format_for(".jpeg") is ie.format_for(".jpg")
    assert ie.format_for(".tiff") is ie.format_for(".tif")


def test_an_unknown_suffix_is_reported_rather_than_guessed():

    assert ie.format_for(".docx") is None


def test_only_the_alpha_raster_formats_offer_transparency():

    for fmt in ie.FORMATS:

        expected = fmt.alpha and not fmt.vector

        assert ie.supports_transparency(fmt) is expected, fmt.name


def test_jpeg_and_bmp_do_not_claim_transparency():
    """They have no alpha channel, and asking for one raises."""

    assert not ie.supports_transparency(ie.format_for(".jpg"))
    assert not ie.supports_transparency(ie.format_for(".bmp"))


def test_the_dialog_filter_offers_everything_and_each_alone():

    text = ie.dialog_filter()

    assert text.startswith("Image (")

    for fmt in ie.FORMATS:
        assert fmt.suffix in text


# ----------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------


def test_the_scale_multiplies_both_sides():

    assert ie.output_size((640, 480), 2) == (1280, 960)
    assert ie.output_size((640, 480), 1) == (640, 480)


def test_the_scale_is_held_inside_what_will_render():
    """
    Above about 8x an ordinary viewport is into tens of megapixels and
    the render fails on the GPU rather than producing anything better.
    """

    assert ie.clamp_scale(500) == ie.MAX_SCALE
    assert ie.clamp_scale(0) == ie.MIN_SCALE
    assert ie.clamp_scale(-3) == ie.MIN_SCALE


def test_a_nonsense_scale_falls_back_rather_than_raising():

    assert ie.clamp_scale("wide") == ie.DEFAULT_SCALE
    assert ie.clamp_scale(None) == ie.DEFAULT_SCALE


# ----------------------------------------------------------------
# Naming
# ----------------------------------------------------------------


def test_the_extension_follows_the_chosen_format(tmp_path):

    png = ie.format_for(".png")

    assert ie.with_suffix(tmp_path / "figure.jpg", png).suffix == ".png"
    assert ie.with_suffix(tmp_path / "figure", png).suffix == ".png"


def test_a_matching_extension_is_left_as_it_was(tmp_path):
    """``figure.tiff`` must not become ``figure.tiff.tif``."""

    tiff = ie.format_for(".tif")

    assert ie.with_suffix(tmp_path / "figure.tiff", tiff).name == (
        "figure.tiff"
    )


def test_a_dotted_stem_is_not_mistaken_for_an_extension(tmp_path):

    png = ie.format_for(".png")

    written = ie.with_suffix(tmp_path / "LiCoO2_2x2", png)

    assert written.name == "LiCoO2_2x2.png"


# ----------------------------------------------------------------
# Writing
# ----------------------------------------------------------------


@pytest.mark.parametrize("name", [fmt.name for fmt in ie.FORMATS])
def test_every_offered_format_actually_writes(plotter, tmp_path, name):

    fmt = next(f for f in ie.FORMATS if f.name == name)

    written = ie.save_image(plotter, tmp_path / "figure", fmt, scale=1)

    assert written.exists()
    assert written.stat().st_size > 0
    assert written.suffix == fmt.suffix


def test_the_scale_reaches_the_written_pixels(plotter, tmp_path):

    written = ie.save_image(
        plotter, tmp_path / "big", ie.format_for(".png"), scale=3
    )

    assert opened(written).size == ie.output_size(WINDOW, 3)


def test_a_transparent_png_has_an_alpha_channel(plotter, tmp_path):

    written = ie.save_image(
        plotter,
        tmp_path / "clear",
        ie.format_for(".png"),
        scale=1,
        transparent=True,
    )

    assert opened(written).mode == "RGBA"


def test_asking_a_jpeg_for_transparency_writes_it_opaque(plotter, tmp_path):
    """
    The guard that matters. Passing this through raises OSError from
    inside the image writer -- after the scene has been rendered at full
    scale -- with a message about colour modes and nothing about the
    checkbox responsible.
    """

    written = ie.save_image(
        plotter,
        tmp_path / "opaque",
        ie.format_for(".jpg"),
        scale=1,
        transparent=True,
    )

    assert written.exists()
    assert opened(written).mode == "RGB"


def test_the_same_holds_for_bmp(plotter, tmp_path):

    written = ie.save_image(
        plotter,
        tmp_path / "opaque",
        ie.format_for(".bmp"),
        scale=1,
        transparent=True,
    )

    assert opened(written).mode == "RGB"


def test_a_vector_export_ignores_the_scale(plotter, tmp_path):
    """There is no resolution to choose, so a scale must not confuse it."""

    small = ie.save_image(
        plotter, tmp_path / "a", ie.format_for(".svg"), scale=1
    )
    large = ie.save_image(
        plotter, tmp_path / "b", ie.format_for(".svg"), scale=8
    )

    assert small.exists() and large.exists()


def test_the_format_is_taken_from_the_name_when_none_is_given(
    plotter, tmp_path
):

    written = ie.save_image(plotter, tmp_path / "guessed.tif", scale=1)

    assert written.suffix == ".tif"


def test_an_unknown_extension_falls_back_to_png(plotter, tmp_path):
    """
    Better than refusing: the user asked for a picture and gets one, in
    the format that was going to be right anyway.
    """

    written = ie.save_image(plotter, tmp_path / "figure.xyz", scale=1)

    assert written.suffix == ".png"
    assert written.exists()


def test_a_missing_folder_is_created(plotter, tmp_path):

    written = ie.save_image(
        plotter,
        tmp_path / "figures" / "new" / "figure.png",
        scale=1,
    )

    assert written.exists()


def test_exporting_does_not_disturb_the_view(plotter, tmp_path):
    """
    The window is rendered again at a multiple of its size. It has to
    come back the size it was, or the viewport is left wrong afterwards.
    """

    before = list(plotter.window_size)

    ie.save_image(plotter, tmp_path / "figure", ie.format_for(".png"), scale=4)

    assert list(plotter.window_size) == before


def test_the_camera_is_untouched_by_an_export(plotter, tmp_path):

    # Export once first. VTK frames the scene lazily on its first real
    # render, so a camera read before that is still at the uninitialised
    # default and would look as though the export had moved it.
    ie.save_image(plotter, tmp_path / "warm", ie.format_for(".png"), scale=1)

    camera = plotter.renderer.GetActiveCamera()

    position = camera.GetPosition()
    focal = camera.GetFocalPoint()

    ie.save_image(plotter, tmp_path / "figure", ie.format_for(".png"), scale=2)

    assert camera.GetPosition() == position
    assert camera.GetFocalPoint() == focal


# ----------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------


@pytest.fixture
def dialog(plotter):

    pytest.importorskip("PySide6.QtWidgets")

    from PySide6.QtWidgets import QApplication

    from qvista.gui.dialogs.image_dialog import ImageDialog

    QApplication.instance() or QApplication([])

    return ImageDialog(plotter, "LiCoO2")


def test_the_dialog_offers_every_format(dialog):

    assert dialog.format_box.count() == len(ie.FORMATS)


def test_a_vector_format_switches_off_the_resolution(dialog):
    """There is nothing to choose, so it should not look choosable."""

    for index in range(dialog.format_box.count()):

        dialog.format_box.setCurrentIndex(index)

        assert dialog.scale_box.isEnabled() is (
            not dialog.current_format().vector
        )


def test_a_format_without_alpha_switches_off_transparency(dialog):

    for index in range(dialog.format_box.count()):

        dialog.format_box.setCurrentIndex(index)

        fmt = dialog.current_format()

        assert dialog.transparent.isEnabled() is (
            ie.supports_transparency(fmt)
        )


def test_switching_to_a_format_without_alpha_clears_the_tick(dialog):
    """
    Otherwise the box is unticked on screen while the state behind it
    still says transparent, and the next PNG export silently differs.
    """

    dialog.format_box.setCurrentIndex(0)
    dialog.transparent.setChecked(True)

    jpeg = next(
        i
        for i in range(dialog.format_box.count())
        if dialog.format_box.itemData(i).name == "JPEG"
    )

    dialog.format_box.setCurrentIndex(jpeg)

    assert not dialog.transparent.isChecked()


def test_the_dialog_says_what_the_export_will_come_to(dialog):

    dialog.format_box.setCurrentIndex(0)
    dialog.scale_box.setValue(4)

    width, height = ie.output_size(WINDOW, 4)

    assert f"{width} x {height}" in dialog.size_label.text()


def test_a_vector_format_says_so_instead_of_a_pixel_count(dialog):

    svg = next(
        i
        for i in range(dialog.format_box.count())
        if dialog.format_box.itemData(i).name == "SVG"
    )

    dialog.format_box.setCurrentIndex(svg)

    assert "independent" in dialog.size_label.text()


# ----------------------------------------------------------------
# The orientation marker
# ----------------------------------------------------------------


def corner_ink(path) -> int:
    """
    How much is drawn in the lower-left corner, where the marker sits.

    Counting dark pixels rather than comparing images: the rest of the
    scene is identical between the two exports, so a difference here is
    the marker and nothing else.
    """

    numpy = pytest.importorskip("numpy")

    pixels = numpy.asarray(opened(path).convert("L"), dtype=float)

    height, width = pixels.shape

    corner = pixels[int(height * 0.6) :, : int(width * 0.4)]

    return int((corner < 200).sum())


@pytest.fixture
def marked(plotter):
    """A view with the orientation marker showing."""

    plotter.add_axes(color="#3c4a68")
    plotter.render()

    return plotter


def test_the_marker_can_be_left_out(marked, tmp_path):
    """
    Its labels are 2D captions at a fixed point size while its arrows
    are 3D geometry, so enlarging the render pulls the two apart and the
    letters come out oversized and overlapping.
    """

    png = ie.format_for(".png")

    with_axes = ie.save_image(
        marked, tmp_path / "with", png, scale=2, axes=True
    )
    without = ie.save_image(
        marked, tmp_path / "without", png, scale=2, axes=False
    )

    assert corner_ink(without) < corner_ink(with_axes)


def test_the_marker_comes_back_after_an_export(marked, tmp_path):
    """Hiding it for a figure must not cost the user their axes."""

    widget = marked.renderer.axes_widget

    assert widget.GetEnabled()

    ie.save_image(
        marked, tmp_path / "figure", ie.format_for(".png"), axes=False
    )

    assert widget.GetEnabled()


def test_the_marker_comes_back_even_when_the_export_fails(marked, tmp_path):
    """
    The restore is in a ``finally``. A failed export leaving the viewport
    changed would be a second, silent failure on top of the first.
    """

    widget = marked.renderer.axes_widget

    class Doomed:
        renderer = marked.renderer

        def screenshot(self, *_args, **_kwargs):
            raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        ie.save_image(
            Doomed(), tmp_path / "figure", ie.format_for(".png"), axes=False
        )

    assert widget.GetEnabled()


def test_keeping_the_marker_touches_nothing(marked, tmp_path):

    widget = marked.renderer.axes_widget

    ie.save_image(
        marked, tmp_path / "figure", ie.format_for(".png"), axes=True
    )

    assert widget.GetEnabled()


def test_a_view_with_no_marker_exports_anyway(plotter, tmp_path):
    """Nothing to hide is not an error."""

    written = ie.save_image(
        plotter, tmp_path / "figure", ie.format_for(".png"), axes=False
    )

    assert written.exists()


def test_the_dialog_leaves_the_marker_out_by_default(dialog):
    """
    The oversized-label problem is the common case, so a clean figure
    should be what the first export gives.
    """

    assert not dialog.include_axes.isChecked()
