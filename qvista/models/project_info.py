"""
Data model for a VASP project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectInfo:
    """Stores information about the currently opened project."""

    name: str
    directory: Path

    files: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    #: The structure file the user actually selected, when they picked a
    #: file rather than a directory.
    source_file: Path | None = None
