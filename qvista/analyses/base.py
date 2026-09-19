"""
Common contract for QVista analysis workflows.

Every workflow -- k-path, phonon band, phonon DOS, mechanical properties --
subclasses :class:`Analysis`. The base class owns the parts that are the same
for all of them: precondition checking, structure access, progress reporting
and cancellation. Subclasses implement :meth:`Analysis.execute` only.

Failures are reported as :class:`AnalysisError` with a message written for a
user, in the same spirit as ``StructureAnalysisError``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from qvista.models.project import Project
from qvista.services.structure_analyzer import DEFAULT_SYMPREC

#: Progress callback: ``(message, percent)``. Percent is
#: :data:`INDETERMINATE` when the workflow cannot say how far along it is.
ProgressCallback = Callable[[str, int], None]

#: Percent value meaning "running, but progress is not measurable".
INDETERMINATE = -1


# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------


class AnalysisError(Exception):
    """
    An analysis could not run or could not finish.

    The message is shown directly to the user, so it must explain the
    problem and, where possible, what to do about it.
    """


class AnalysisCancelled(Exception):
    """
    Raised inside a workflow when the user cancels it.

    This is control flow, not a failure, so it deliberately does not
    subclass :class:`AnalysisError` -- the worker reports it separately and
    no error dialog is shown.
    """


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------


@dataclass
class AnalysisResult:
    """
    What a workflow returns.

    ``rows`` are ``(label, value)`` pairs printed to the console in the same
    format as the structure properties block. ``data`` carries the numeric
    payload (band paths, frequencies, tensors) for plotting and for storage
    in :class:`~qvista.models.results.Results`.
    """

    title: str = ""

    rows: list[tuple[str, str]] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    data: dict[str, Any] = field(default_factory=dict)

    def as_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` pairs, for console output."""

        return list(self.rows)


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------


class Analysis(ABC):
    """
    Base class for a long-running analysis workflow.

    Subclasses set :attr:`name`, optionally override :meth:`validate`, and
    implement :meth:`execute`. Inside :meth:`execute` they should call
    :meth:`report` to publish progress and :meth:`raise_if_cancelled` at
    every loop iteration, so cancellation stays responsive.
    """

    #: Shown in the console, the status bar and error dialogs.
    name: str = "Analysis"

    #: Workflows needing a lattice (all of them, so far) leave this True;
    #: :meth:`validate` then rejects plain XYZ input.
    requires_periodic: bool = True

    def __init__(
        self,
        project: Project,
        symprec: float = DEFAULT_SYMPREC,
    ) -> None:

        self.project = project
        self.symprec = symprec

        self._progress: ProgressCallback | None = None
        self._cancelled = False
        self._structure: Any = None

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def run(
        self,
        progress: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """
        Validate preconditions, then execute.

        Called on a worker thread by
        :class:`~qvista.analyses.worker.AnalysisWorker`; it should not be
        called on the GUI thread.
        """

        self._progress = progress
        self._cancelled = False

        self.validate()
        self.raise_if_cancelled()

        return self.execute()

    def validate(self) -> None:
        """
        Check that this analysis can run against the current project.

        Subclasses that need more (a minimum atom count, a specific
        element) should call ``super().validate()`` first.
        """

        # Imported here rather than at module scope: guards imports
        # AnalysisError from this module, so a top-level import would be
        # circular.
        from qvista.analyses.guards import require_periodic

        if self.requires_periodic:
            require_periodic(self.project)

    @abstractmethod
    def execute(self) -> AnalysisResult:
        """Do the work. Implemented by each workflow."""

    # ------------------------------------------------------------
    # Structure access
    # ------------------------------------------------------------

    def structure(self) -> Any:
        """
        Return the project's structure as a pymatgen object.

        Always routed through
        :meth:`~qvista.services.structure_analyzer.StructureAnalyzer.parse`,
        which is the only path that handles XYZ correctly. The result is
        cached, so repeated calls inside a loop are free.
        """

        if self._structure is not None:
            return self._structure

        from qvista.services.structure_analyzer import (
            StructureAnalysisError,
            StructureAnalyzer,
        )

        source = self.project.info.source_file

        if source is None:
            raise AnalysisError(
                "This project has no structure file. "
                "Open a POSCAR, CONTCAR, CIF or XYZ file first."
            )

        try:
            parsed = StructureAnalyzer.parse(source)

        except StructureAnalysisError as exc:
            raise AnalysisError(str(exc)) from exc

        self._structure = parsed.structure

        return self._structure

    # ------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------

    def report(
        self,
        message: str,
        percent: int = INDETERMINATE,
    ) -> None:
        """Publish a progress message, if anyone is listening."""

        if self._progress is not None:
            self._progress(message, percent)

    def report_step(
        self,
        index: int,
        total: int,
        message: str,
    ) -> None:
        """
        Publish progress through a loop of ``total`` steps.

        ``index`` is zero-based. Workflows that loop over hundreds of
        displaced structures use this.
        """

        if total <= 0:
            self.report(message)
            return

        percent = int(round(100.0 * (index + 1) / total))

        self.report(
            f"{message}  ({index + 1}/{total})",
            min(percent, 100),
        )

    # ------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------

    def cancel(self) -> None:
        """
        Ask the workflow to stop.

        Called from the GUI thread while :meth:`execute` runs on the worker
        thread; it only sets a flag, which the workflow observes at its next
        :meth:`raise_if_cancelled` call.
        """

        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""

        return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Abort the workflow if cancellation has been requested."""

        if self._cancelled:
            raise AnalysisCancelled(f"{self.name} cancelled")
