"""
Camera navigation shared by every 3D view.

VTK's trackball style zooms down the camera axis: the wheel moves the
camera towards the centre of the viewport, wherever the pointer happens
to be. That is the wrong behaviour for looking at a structure. Anything
worth a closer look is off-centre by definition, so zooming in pushes it
further towards the edge and out of frame, and getting to it means
alternating wheel and drag.

:func:`enable_zoom_to_cursor` makes the wheel zoom towards the pointer
instead, so the atom under the cursor stays under the cursor. That is
one line of intent and three of arithmetic:

1. Find the world point under the pointer, on the plane through the
   focal point.
2. Zoom.
3. Find where that same display position now lands, and slide the camera
   sideways by the difference.

Step 3 is what separates this from a plain dolly -- without it the view
scales about its centre and the point drifts.
"""

from __future__ import annotations

#: Zoom per wheel notch. VTK's own default works out to about 1.2 per
#: notch, which overshoots on a small cell; this is gentler and still
#: crosses a decade in a dozen notches.
ZOOM_STEP = 1.12

#: The camera may not be dollied closer to its focal point than this, in
#: Angstrom. At zero the view direction becomes undefined and the scene
#: turns inside out; there is nothing to see at 1e-6 A anyway.
MIN_DISTANCE = 1.0e-6


def _world_at(renderer, x: float, y: float, through) -> tuple | None:
    """
    The world point under display position ``(x, y)``.

    A display position names a ray, not a point, so one has to be chosen
    along it: this takes the plane parallel to the screen that passes
    through ``through`` -- the focal point in practice, which is the
    depth the view is scaling about.

    Display coordinates are VTK's, with the origin at the bottom left.
    """

    renderer.SetWorldPoint(through[0], through[1], through[2], 1.0)
    renderer.WorldToDisplay()

    depth = renderer.GetDisplayPoint()[2]

    renderer.SetDisplayPoint(float(x), float(y), depth)
    renderer.DisplayToWorld()

    wx, wy, wz, w = renderer.GetWorldPoint()

    # A homogeneous w of zero is a point at infinity -- it happens when
    # the camera is degenerate, and there is nothing sensible to return.
    if w == 0.0:
        return None

    return (wx / w, wy / w, wz / w)


def zoom_at_cursor(plotter, factor: float, position=None) -> None:
    """
    Zoom by ``factor`` about the pointer rather than the view centre.

    ``position`` is a display ``(x, y)``; the current event position is
    used when it is not given, which is what the wheel handler wants.
    """

    interactor = plotter.iren.interactor

    if position is None:
        position = interactor.GetEventPosition()

    x, y = position

    # In a single-renderer view this is plotter.renderer, but asking
    # which renderer is under the pointer is both cheap and correct.
    renderer = interactor.FindPokedRenderer(int(x), int(y))

    if renderer is None:
        renderer = plotter.renderer

    camera = renderer.GetActiveCamera()

    focal = camera.GetFocalPoint()

    before = _world_at(renderer, x, y, focal)

    # ----------------------------------------------------------
    # Zoom
    # ----------------------------------------------------------

    if camera.GetParallelProjection():

        # There is no distance to change under parallel projection; the
        # height of the view is the whole of the scale.
        camera.SetParallelScale(camera.GetParallelScale() / factor)

    else:

        if camera.GetDistance() / factor < MIN_DISTANCE:
            return

        camera.Dolly(factor)

    # ----------------------------------------------------------
    # Slide, so the point under the pointer did not move
    # ----------------------------------------------------------

    after = _world_at(renderer, x, y, camera.GetFocalPoint())

    if before is not None and after is not None:

        shift = [b - a for b, a in zip(before, after)]

        camera.SetPosition(
            *[value + move for value, move in zip(camera.GetPosition(), shift)]
        )

        camera.SetFocalPoint(
            *[value + move for value, move in zip(focal, shift)]
        )

    renderer.ResetCameraClippingRange()

    plotter.render()


def mute_style_wheel(plotter) -> None:
    """
    Stop the current interactor style from acting on the wheel itself.

    Its handler multiplies its dolly by ``MouseWheelMotionFactor``, so a
    factor of zero leaves a dolly of exactly 1 and the camera untouched,
    and :func:`zoom_at_cursor` then does the whole job. Aborting the
    event instead would work too, but only until something else wants
    the wheel as well.

    Separate from :func:`enable_zoom_to_cursor` because the factor lives
    on the style object, not the interactor: swapping the style -- which
    is how the drag mode remaps the left button -- installs a fresh one
    at its default factor, and the wheel would start zooming twice per
    notch. Anything that changes the style calls this afterwards.
    """

    style = plotter.iren.interactor.GetInteractorStyle()

    if hasattr(style, "SetMouseWheelMotionFactor"):
        style.SetMouseWheelMotionFactor(0.0)


def enable_zoom_to_cursor(plotter, step: float = ZOOM_STEP) -> list:
    """
    Point the mouse wheel at the cursor, in ``plotter``.

    Returns the observer handles, so a caller that needs to undo this
    can. The observers go on the interactor rather than the style, so
    they survive a style swap; the style's own wheel handling is muted
    by :func:`mute_style_wheel`.
    """

    mute_style_wheel(plotter)

    return [
        plotter.iren.add_observer(
            "MouseWheelForwardEvent",
            lambda *_args: zoom_at_cursor(plotter, step),
        ),
        plotter.iren.add_observer(
            "MouseWheelBackwardEvent",
            lambda *_args: zoom_at_cursor(plotter, 1.0 / step),
        ),
    ]
