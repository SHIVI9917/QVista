"""
Tests for zooming towards the cursor.

These render. Everything else in the suite is pure, so the whole module
steps aside on a machine that cannot open a render window rather than
failing there.
"""

from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista")

from qvista.visualization.navigation import (  # noqa: E402
    MIN_DISTANCE,
    ZOOM_STEP,
    _world_at,
    enable_zoom_to_cursor,
    zoom_at_cursor,
)

#: Well away from the centre of the view -- the whole point is what
#: happens off-centre, and a bug that only scales about the middle would
#: pass a centred test.
CURSOR = (120, 380)

#: The middle of the window below, where this should behave like a plain
#: dolly because there is nothing to correct.
CENTRE = (320, 240)

WINDOW = (640, 480)


@pytest.fixture
def plotter():
    """An off-screen view of a sphere, looked at from an angle."""

    try:
        view = pv.Plotter(off_screen=True, window_size=WINDOW)

    except Exception as exc:  # pragma: no cover - no GL available
        pytest.skip(f"no render window: {exc}")

    view.add_mesh(pv.Sphere(radius=2.0))
    view.view_isometric()
    view.reset_camera()
    view.render()

    yield view

    view.close()


def camera(view):
    return view.renderer.GetActiveCamera()


def under(view, position=CURSOR):
    """The world point currently under a display position."""

    return _world_at(
        view.renderer, position[0], position[1], camera(view).GetFocalPoint()
    )


def distance_between(a, b) -> float:

    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# ----------------------------------------------------------------
# The invariant
# ----------------------------------------------------------------


def test_the_point_under_the_cursor_does_not_move_zooming_in(plotter):

    before = under(plotter)

    for _ in range(5):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert distance_between(before, under(plotter)) < 1e-6


def test_the_point_under_the_cursor_does_not_move_zooming_out(plotter):

    before = under(plotter)

    for _ in range(5):
        zoom_at_cursor(plotter, 1.0 / ZOOM_STEP, position=CURSOR)

    assert distance_between(before, under(plotter)) < 1e-6


def test_a_plain_dolly_is_what_this_is_fixing(plotter):
    """The behaviour being replaced, so the test above means something."""

    before = under(plotter)

    for _ in range(5):
        camera(plotter).Dolly(ZOOM_STEP)
        plotter.renderer.ResetCameraClippingRange()

    # Not a small drift: the better part of an Angstrom, on a 2 A sphere.
    assert distance_between(before, under(plotter)) > 0.5


def test_it_really_zooms(plotter):
    """Holding the point still must not have cost the zoom."""

    start = camera(plotter).GetDistance()

    for _ in range(5):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert camera(plotter).GetDistance() == pytest.approx(
        start / ZOOM_STEP**5, rel=1e-6
    )


def test_one_notch_is_exactly_the_step(plotter):

    start = camera(plotter).GetDistance()

    zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert start / camera(plotter).GetDistance() == pytest.approx(ZOOM_STEP)


def test_in_then_out_returns_the_camera(plotter):
    """Reversible, so exploring cannot slowly lose the view."""

    position = camera(plotter).GetPosition()
    focal = camera(plotter).GetFocalPoint()

    for _ in range(6):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    for _ in range(6):
        zoom_at_cursor(plotter, 1.0 / ZOOM_STEP, position=CURSOR)

    assert distance_between(position, camera(plotter).GetPosition()) < 1e-6
    assert distance_between(focal, camera(plotter).GetFocalPoint()) < 1e-6


def test_zooming_at_the_centre_leaves_the_focal_point_alone(plotter):
    """At the centre there is nothing to correct, so nothing should be."""

    focal = camera(plotter).GetFocalPoint()

    for _ in range(4):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CENTRE)

    assert distance_between(focal, camera(plotter).GetFocalPoint()) < 1e-6


def test_zooming_off_centre_does_move_the_focal_point(plotter):
    """...and away from it, the camera has to travel sideways."""

    focal = camera(plotter).GetFocalPoint()

    for _ in range(4):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert distance_between(focal, camera(plotter).GetFocalPoint()) > 0.1


# ----------------------------------------------------------------
# Parallel projection
# ----------------------------------------------------------------


def test_parallel_projection_zooms_by_scale(plotter):
    """There is no distance to change, so the view height carries it."""

    camera(plotter).SetParallelProjection(True)
    plotter.render()

    start = camera(plotter).GetParallelScale()

    zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert camera(plotter).GetParallelScale() == pytest.approx(
        start / ZOOM_STEP
    )


def test_parallel_projection_holds_the_point_too(plotter):

    camera(plotter).SetParallelProjection(True)
    plotter.render()

    before = under(plotter)

    for _ in range(4):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert distance_between(before, under(plotter)) < 1e-6


# ----------------------------------------------------------------
# Guards
# ----------------------------------------------------------------


def test_the_camera_cannot_be_dollied_into_its_focal_point(plotter):
    """At zero distance the view direction is undefined and the scene
    turns inside out, so zooming has to stop short of it."""

    view = camera(plotter)

    focal = view.GetFocalPoint()
    direction = view.GetDirectionOfProjection()

    # Just inside the floor.
    tiny = MIN_DISTANCE * 1.05

    view.SetPosition(
        *[f - d * tiny for f, d in zip(focal, direction)]
    )

    position = view.GetPosition()

    zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert view.GetPosition() == position


def test_zooming_out_from_the_floor_still_works(plotter):
    """The guard is one-way: it must not trap the camera."""

    view = camera(plotter)

    focal = view.GetFocalPoint()
    direction = view.GetDirectionOfProjection()

    view.SetPosition(
        *[f - d * (MIN_DISTANCE * 1.05) for f, d in zip(focal, direction)]
    )

    zoom_at_cursor(plotter, 1.0 / ZOOM_STEP, position=CURSOR)

    assert view.GetDistance() > MIN_DISTANCE


# ----------------------------------------------------------------
# Arming it
# ----------------------------------------------------------------


def test_enabling_switches_off_the_styles_own_wheel_zoom(plotter):
    """
    Otherwise both fire and the view jumps two steps per notch, one of
    them towards the middle.
    """

    style = plotter.iren.interactor.GetInteractorStyle()

    assert style.GetMouseWheelMotionFactor() != 0.0

    enable_zoom_to_cursor(plotter)

    assert style.GetMouseWheelMotionFactor() == 0.0


def test_enabling_returns_both_wheel_observers(plotter):

    assert len(enable_zoom_to_cursor(plotter)) == 2


def test_the_wheel_event_zooms_in(plotter):

    enable_zoom_to_cursor(plotter)

    start = camera(plotter).GetDistance()

    plotter.iren.interactor.SetEventPosition(*CURSOR)
    plotter.iren.interactor.InvokeEvent("MouseWheelForwardEvent")

    assert start / camera(plotter).GetDistance() == pytest.approx(
        ZOOM_STEP, rel=1e-3
    )


def test_the_wheel_event_zooms_out(plotter):

    enable_zoom_to_cursor(plotter)

    start = camera(plotter).GetDistance()

    plotter.iren.interactor.SetEventPosition(*CURSOR)
    plotter.iren.interactor.InvokeEvent("MouseWheelBackwardEvent")

    assert camera(plotter).GetDistance() / start == pytest.approx(
        ZOOM_STEP, rel=1e-3
    )


def test_the_wheel_holds_the_point_under_the_cursor(plotter):
    """End to end, through the event rather than the function."""

    enable_zoom_to_cursor(plotter)

    plotter.iren.interactor.SetEventPosition(*CURSOR)

    before = under(plotter)

    for _ in range(4):
        plotter.iren.interactor.InvokeEvent("MouseWheelForwardEvent")

    assert distance_between(before, under(plotter)) < 1e-6


# ----------------------------------------------------------------
# Keeping the wheel muted across a style swap
# ----------------------------------------------------------------


def wheel_factor(view) -> float:
    """How much the current style dollies per wheel notch."""

    return view.iren.interactor.GetInteractorStyle().GetMouseWheelMotionFactor()


def test_swapping_the_style_brings_its_wheel_back(plotter):
    """
    Not desired behaviour but the hazard behind :func:`mute_style_wheel`.

    The factor lives on the style object. Installing a new one -- which
    is how the drag mode remaps the left button -- arrives at VTK's
    default, while the zoom-to-cursor observers sit on the interactor
    and survive. Both would then fire, two steps per notch.
    """

    enable_zoom_to_cursor(plotter)

    assert wheel_factor(plotter) == 0.0

    plotter.enable_custom_trackball_style(left="pan")

    assert wheel_factor(plotter) != 0.0


def test_muting_survives_a_style_swap(plotter):

    from qvista.visualization.navigation import mute_style_wheel

    enable_zoom_to_cursor(plotter)

    plotter.enable_custom_trackball_style(left="pan")
    mute_style_wheel(plotter)

    assert wheel_factor(plotter) == 0.0


def test_muting_a_style_without_a_wheel_setting_is_harmless(plotter):
    """Not every VTK style exposes the factor."""

    from qvista.visualization.navigation import mute_style_wheel

    plotter.enable_image_style()

    mute_style_wheel(plotter)


def test_the_cursor_still_holds_after_a_style_swap(plotter):
    """
    The two features have to coexist: entering drag mode must not cost
    the zoom its anchor.
    """

    from qvista.visualization.navigation import mute_style_wheel

    enable_zoom_to_cursor(plotter)

    plotter.enable_custom_trackball_style(left="pan")
    mute_style_wheel(plotter)

    before = under(plotter)

    for _ in range(4):
        zoom_at_cursor(plotter, ZOOM_STEP, position=CURSOR)

    assert distance_between(before, under(plotter)) < 1e-6
