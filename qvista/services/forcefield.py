"""
Optional machine-learning force field.

Forces and stresses for the phonon and mechanical-property workflows come
from a machine-learning interatomic potential. These are **not** DFT: they
are fast enough to evaluate hundreds of displaced supercells interactively,
but their accuracy is that of a trained surrogate model, and results should
not be quoted as DFT-quality.

Several potentials are supported, and the first one installed is used. That
matters in practice: MatterSim publishes no wheels, so on Windows it can
only be installed with the Microsoft C++ Build Tools present, whereas CHGNet
and MACE ship wheels and install anywhere.

Nothing here is imported at module scope, so QVista starts and runs normally
with no force field installed at all; the workflows that need one report
that it is missing instead of failing at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class ForceFieldUnavailable(Exception):
    """Raised when a force field is needed but none is installed."""


# ------------------------------------------------------------
# Backends
# ------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """One supported machine-learning potential."""

    key: str
    name: str

    #: Module probed to decide whether it is installed.
    module: str

    #: What to run to install it.
    install: str

    #: Shown in the console and stamped on figures.
    description: str

    #: Builds the ASE calculator. Imports happen inside.
    factory: Callable

    #: Builds a second calculator around an existing one's loaded
    #: model. ``None`` where the backend offers no way to, and every
    #: caller then loads its own copy of the weights.
    sibling: Callable | None = None

    def is_installed(self) -> bool:
        """True if the backend's package can be imported."""

        import importlib.util

        return importlib.util.find_spec(self.module) is not None


def _mattersim():
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator()


#: Where a chosen CHGNet model file is remembered, set from
#: File > Set CHGNet Path. Empty means the weights bundled with the
#: installed package.
CHGNET_PATH_KEY = "forcefield/chgnet_model"


def chgnet_model_path() -> str:
    """
    The model file the user chose, or ``""`` for the bundled default.

    Read at every call rather than cached, so changing it in the File
    menu takes effect on the next analysis instead of the next session.
    """

    try:
        from PySide6.QtCore import QSettings

        from qvista.core.config import APP_NAME, ORGANIZATION

        return str(
            QSettings(ORGANIZATION, APP_NAME).value(
                CHGNET_PATH_KEY, "", type=str
            )
            or ""
        ).strip()

    except Exception:  # pragma: no cover - no Qt, e.g. in a script
        return ""


def _chgnet():
    try:
        from chgnet.model.dynamics import CHGNetCalculator

    except ImportError:  # pragma: no cover - depends on the version
        from chgnet.model import CHGNetCalculator

    chosen = chgnet_model_path()

    if not chosen:
        return CHGNetCalculator()

    from pathlib import Path

    path = Path(chosen)

    if not path.is_file():
        raise ForceFieldUnavailable(
            f"The CHGNet model set in File > Set CHGNet Path is not "
            f"there:\n\n    {path}\n\nClear the setting to fall back to "
            f"the weights bundled with the installed package."
        )

    from chgnet.model import CHGNet

    try:
        model = CHGNet.from_file(str(path))

    except Exception as exc:
        raise ForceFieldUnavailable(
            f"{path.name} could not be loaded as a CHGNet model: "
            f"{type(exc).__name__}: {exc}\n\nIt should be a .pth.tar "
            f"file saved by CHGNet -- a fine-tuned model, or one of the "
            f"pretrained sets."
        ) from exc

    return CHGNetCalculator(model=model)


def _chgnet_sibling(calculator):
    """Another CHGNet calculator around an already-loaded model."""

    return type(calculator)(model=calculator.model)


def _mace():
    from mace.calculators import mace_mp

    return mace_mp(default_dtype="float64")


#: Tried in this order. MatterSim first because it is what the workflows
#: were originally written against; the others are the installable
#: fallbacks.
BACKENDS: tuple[Backend, ...] = (
    Backend(
        key="mattersim",
        name="MatterSim",
        module="mattersim",
        install="pip install mattersim",
        description="MatterSim (machine-learning force field - not DFT)",
        factory=_mattersim,
    ),
    Backend(
        key="chgnet",
        name="CHGNet",
        module="chgnet",
        install="pip install chgnet",
        description="CHGNet (machine-learning force field - not DFT)",
        factory=_chgnet,
        sibling=_chgnet_sibling,
    ),
    Backend(
        key="mace",
        name="MACE-MP",
        module="mace",
        install="pip install mace-torch",
        description="MACE-MP (machine-learning force field - not DFT)",
        factory=_mace,
    ),
)


# ------------------------------------------------------------
# Availability
# ------------------------------------------------------------


def available_backends() -> list[Backend]:
    """Every supported potential that is currently installed."""

    return [backend for backend in BACKENDS if backend.is_installed()]


def active_backend() -> Backend | None:
    """The potential that will be used, or ``None`` if none is installed."""

    installed = available_backends()

    return installed[0] if installed else None


def is_available() -> bool:
    """True when at least one force field is installed."""

    return active_backend() is not None


def describe() -> str:
    """
    One-line label for console output and figure annotations.

    Names the potential that actually ran, so a result can never be
    attributed to the wrong model.
    """

    backend = active_backend()

    if backend is None:
        return "no force field installed"

    # A chosen model file is a different model, so it has to appear in
    # the label. A result from a fine-tuned network attributed to plain
    # CHGNet is exactly the misattribution this function exists to stop.
    if backend.key == "chgnet":

        chosen = chgnet_model_path()

        if chosen:
            from pathlib import Path

            return f"{backend.description} [{Path(chosen).name}]"

    return backend.description


def unavailable_message() -> str:
    """Explain, for a user, that no force field is installed."""

    lines = [
        "No machine-learning force field is installed, so forces cannot "
        "be evaluated.",
        "",
        "Install any one of these:",
        "",
    ]

    for backend in BACKENDS:
        lines.append(f"    {backend.install:<28} # {backend.name}")

    lines.extend(
        [
            "",
            "CHGNet and MACE ship prebuilt wheels and install anywhere.",
            "MatterSim has no wheels, so on Windows it additionally needs "
            "the Microsoft C++ Build Tools.",
            "",
            "All of them depend on torch, which is a large download. "
            "Everything else in QVista works without one.",
        ]
    )

    return "\n".join(lines)


# ------------------------------------------------------------
# Calculator
# ------------------------------------------------------------


#: One loaded calculator per backend, kept to build later ones around.
#: The network is expensive to load and stateless once loaded; the ASE
#: wrapper around it is cheap and stateful, holding the last atoms it
#: saw and their results. Sharing the first and not the second is what
#: lets a band of images each keep its own cache without paying to load
#: the weights again per image.
_TEMPLATES: dict[str, object] = {}


def forget_calculators() -> None:
    """Drop the cached models, so the next call loads afresh."""

    _TEMPLATES.clear()


def get_calculator(key: str | None = None):
    """
    Return an ASE calculator from the requested or first-available backend.

    Each call returns a *separate* calculator object, though where the
    backend allows it they share one copy of the loaded weights. That
    distinction matters to anything holding several structures at once:
    ASE caches a calculator's results against the last atoms it was
    given, so one calculator shared between images is an eviction on
    every turn. On a 47-atom NEB that came to 11 force evaluations per
    optimiser step where 2 were needed.

    Raises
    ------
    ForceFieldUnavailable
        If nothing suitable is installed, with installation instructions.
    """

    if key is None:
        backend = active_backend()

    else:
        backend = next(
            (item for item in BACKENDS if item.key == key),
            None,
        )

        if backend is None:
            raise ForceFieldUnavailable(f"Unknown force field: {key}")

        if not backend.is_installed():
            raise ForceFieldUnavailable(
                f"{backend.name} is not installed.\n\n"
                f"Install it with:\n    {backend.install}"
            )

    if backend is None:
        raise ForceFieldUnavailable(unavailable_message())

    template = _TEMPLATES.get(backend.key)

    if template is not None and backend.sibling is not None:

        try:
            return backend.sibling(template)

        except Exception:  # pragma: no cover - depends on the version
            # The cached one is no use to us; drop it and load again
            # below rather than fail over a shortcut.
            _TEMPLATES.pop(backend.key, None)

    try:
        calculator = backend.factory()

    except ForceFieldUnavailable:
        # The factory already explained itself -- a missing model file,
        # say. Wrapping it again buries that message inside a generic
        # one that says less.
        raise

    except Exception as exc:  # pragma: no cover - depends on the install
        raise ForceFieldUnavailable(
            f"{backend.name} is installed but could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if backend.sibling is not None:
        _TEMPLATES.setdefault(backend.key, calculator)

    return calculator
