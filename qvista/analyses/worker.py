"""
Worker thread for running an analysis off the GUI thread.

Phonon and elastic workflows loop over hundreds of displaced structures and
take minutes to hours. Running one on the GUI thread would freeze the whole
window, so :class:`AnalysisWorker` runs it on a ``QThread`` and reports back
through signals, which Qt delivers on the GUI thread.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from qvista.analyses.base import (
    Analysis,
    AnalysisCancelled,
    AnalysisError,
    AnalysisResult,
)


class AnalysisWorker(QThread):
    """
    Runs one :class:`~qvista.analyses.base.Analysis` to completion.

    Exactly one of :attr:`completed`, :attr:`failed` or :attr:`cancelled` is
    emitted per run. ``QThread`` already defines ``started`` and
    ``finished``, so the completion signal here is named ``completed``.
    """

    #: ``(message, percent)``; percent is -1 when not measurable.
    progress = Signal(str, int)

    #: Emitted with the :class:`AnalysisResult` on success.
    completed = Signal(object)

    #: Emitted with a user-readable message on failure.
    failed = Signal(str)

    #: Emitted when the user cancelled the run.
    cancelled = Signal()

    def __init__(
        self,
        analysis: Analysis,
        parent=None,
    ) -> None:

        super().__init__(parent)

        self.analysis = analysis

    # ------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the analysis. Runs on the worker thread.

        Every exception is converted into a signal; nothing is allowed to
        escape, because an exception crossing a ``QThread`` boundary
        terminates the application.
        """

        try:
            result = self.analysis.run(progress=self._report)

        except AnalysisCancelled:
            self.cancelled.emit()
            return

        except AnalysisError as exc:
            self.failed.emit(str(exc))
            return

        except Exception as exc:
            # A bug rather than a handled condition; still must not escape.
            self.failed.emit(
                f"{self.analysis.name} failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        self.completed.emit(result)

    # ------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------

    def _report(
        self,
        message: str,
        percent: int,
    ) -> None:
        """
        Forward a progress update from the analysis.

        Called on the worker thread; emitting a signal is how the value
        reaches the GUI thread safely.
        """

        self.progress.emit(message, percent)

    # ------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------

    def cancel(self) -> None:
        """
        Ask the running analysis to stop.

        Called from the GUI thread. The analysis stops at its next
        cancellation check, so this returns immediately rather than
        blocking until the thread has actually finished.
        """

        self.analysis.cancel()

        self.requestInterruption()


def result_or_none(value: object) -> AnalysisResult | None:
    """
    Narrow the payload of :attr:`AnalysisWorker.completed` to a result.

    The signal is typed ``object`` because Qt cannot carry an arbitrary
    Python class directly; this keeps the cast in one place.
    """

    if isinstance(value, AnalysisResult):
        return value

    return None
