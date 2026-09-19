"""
Handing out calculators without reloading the network.

The expensive half of a machine-learning calculator is the model; the
cheap half is the ASE wrapper, which caches its last result against the
atoms it was given. Anything holding several structures at once wants
its own wrapper per structure and one shared copy of the weights, and
these tests pin that down.
"""

from __future__ import annotations

import pytest

from qvista.services import forcefield


class FakeModel:
    """Stands in for the loaded network."""


class FakeCalculator:
    """Stands in for the ASE wrapper around one."""

    def __init__(self, model=None) -> None:
        self.model = model if model is not None else FakeModel()


@pytest.fixture
def counted(monkeypatch):
    """A backend that shares its model, and a count of real loads."""

    loads = {"n": 0}

    def factory():
        loads["n"] += 1
        return FakeCalculator()

    backend = forcefield.Backend(
        key="fake",
        name="Fake",
        module="pathlib",  # importable, so it counts as installed
        install="pip install nothing",
        description="Fake",
        factory=factory,
        sibling=lambda calculator: FakeCalculator(model=calculator.model),
    )

    monkeypatch.setattr(forcefield, "active_backend", lambda: backend)

    forcefield.forget_calculators()

    yield loads

    forcefield.forget_calculators()


def test_each_call_returns_a_separate_calculator(counted) -> None:
    """
    Two callers must not share one wrapper.

    This is the whole point: ASE evicts a calculator's cached result
    when it is handed different atoms, so one wrapper shared across a
    band of images recomputes on every image's turn.
    """

    first = forcefield.get_calculator()
    second = forcefield.get_calculator()

    assert first is not second


def test_separate_calculators_share_one_model(counted) -> None:
    """The weights are loaded once, however many wrappers are asked for."""

    calculators = [forcefield.get_calculator() for _ in range(5)]

    models = {id(item.model) for item in calculators}

    assert len(models) == 1
    assert counted["n"] == 1


def test_a_backend_without_a_sibling_loads_every_time(monkeypatch) -> None:
    """
    Nothing is cached for a backend that cannot share.

    Handing the same wrapper out twice would be worse than loading
    twice, so the absence of a sibling means no caching at all.
    """

    loads = {"n": 0}

    def factory():
        loads["n"] += 1
        return FakeCalculator()

    backend = forcefield.Backend(
        key="lonely",
        name="Lonely",
        module="pathlib",
        install="pip install nothing",
        description="Lonely",
        factory=factory,
    )

    monkeypatch.setattr(forcefield, "active_backend", lambda: backend)

    forcefield.forget_calculators()

    first = forcefield.get_calculator()
    second = forcefield.get_calculator()

    assert first is not second
    assert loads["n"] == 2


def test_a_broken_sibling_falls_back_to_loading(monkeypatch) -> None:
    """
    A shortcut that fails costs time, not the run.

    The sibling call depends on a third-party constructor keeping its
    signature, so a version that changed it should mean a slow
    calculator rather than no calculator.
    """

    loads = {"n": 0}

    def factory():
        loads["n"] += 1
        return FakeCalculator()

    def broken(calculator):
        raise TypeError("unexpected keyword argument 'model'")

    backend = forcefield.Backend(
        key="broken",
        name="Broken",
        module="pathlib",
        install="pip install nothing",
        description="Broken",
        factory=factory,
        sibling=broken,
    )

    monkeypatch.setattr(forcefield, "active_backend", lambda: backend)

    forcefield.forget_calculators()

    first = forcefield.get_calculator()
    second = forcefield.get_calculator()

    assert isinstance(second, FakeCalculator)
    assert first is not second
    assert loads["n"] == 2

    forcefield.forget_calculators()


def test_forgetting_reloads(counted) -> None:
    """Dropping the cache means the next call loads afresh."""

    forcefield.get_calculator()
    assert counted["n"] == 1

    forcefield.forget_calculators()

    forcefield.get_calculator()
    assert counted["n"] == 2


def test_a_missing_backend_still_raises(monkeypatch) -> None:
    """Caching does not swallow the "nothing installed" message."""

    monkeypatch.setattr(forcefield, "active_backend", lambda: None)

    forcefield.forget_calculators()

    with pytest.raises(forcefield.ForceFieldUnavailable):
        forcefield.get_calculator()
