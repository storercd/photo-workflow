"""
Extension interface for Live pipeline stages.

A feature session adds one module, `photo_workflow/live/features/<name>.py`, that
defines `register(registry: StageRegistry) -> None` and calls
`registry.register(StageId.<STAGE>, MyStage())`. The framework discovers every module
in `photo_workflow.live.features` at start-up, so features never edit the framework or
each other's files. See `docs/live-mode.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import SPECS, StageId, StageStatus


class PermanentStageError(Exception):
    """Raised by a stage when retrying cannot help; the feature halts immediately."""


@dataclass(frozen=True)
class LiveWarning:
    """A non-fatal problem (for example an interrupted upload) the user can inspect."""

    id: int
    stage: StageId
    message: str
    photo: Path | None = None
    created: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class StageResult:
    """
    Outcome of processing one photo.

    Attributes:
        status: `DONE` when the stage produced output, `SKIPPED` when it had nothing to
            do (for example no subject was found) and the input continues downstream.
        outputs: Files written, least to most advanced. The last one is what later
            stages and the viewer use as their source.
        message: Optional note shown in logs and tooltips.
    """

    status: StageStatus
    outputs: tuple[Path, ...] = ()
    message: str = ""

    @classmethod
    def done(cls, *outputs: Path, message: str = "") -> StageResult:
        """Return a successful result listing the files written."""
        return cls(StageStatus.DONE, tuple(outputs), message)

    @classmethod
    def skipped(cls, message: str = "") -> StageResult:
        """Return a result for a stage that had nothing to do."""
        return cls(StageStatus.SKIPPED, (), message)


@dataclass(frozen=True)
class StageJob:
    """
    Everything a stage needs to process one photo.

    Attributes:
        stage: The stage being run.
        photo: The renamed original in the date folder's `input` subfolder; its XMP
            sidecar is `photo.with_suffix(".xmp")`.
        source: The most advanced earlier output available (skipping stages that are
            off, missing or skipped), falling back to `photo` itself.
        source_stage: The stage that produced `source`, or `None` for the original.
        layout: Paths inside the Live folder.
        settings: A snapshot of the Live settings taken when the job started.
        warn: Reports a non-fatal warning for this photo.
    """

    stage: StageId
    photo: Path
    source: Path
    source_stage: StageId | None
    layout: LiveLayout
    settings: LiveSettings
    warn: Callable[[str], None]

    def output_path(self, subfolder: str | None = None, suffix: str | None = None) -> Path:
        """
        Return where to write this stage's output, creating its folder.

        `subfolder` defaults to the stage's primary subfolder. Existing files are
        overwritten, which is how regenerated output replaces older output.

        Returns:
            The path of the file to write.
        """
        name = subfolder or SPECS[self.stage].subfolders[0]
        path = self.layout.output_path(self.photo, name, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class StageHost(Protocol):
    """What the framework offers a stage's `start` hook (for example the FTP server)."""

    @property
    def layout(self) -> LiveLayout:
        """The current Live folder layout."""

    @property
    def settings(self) -> LiveSettings:
        """The current Live settings."""

    def ingest(self, path: Path) -> None:
        """Hand over a completed upload; it is renamed first, then enters the pipeline."""

    def warn(self, stage: StageId, message: str, photo: Path | None = None) -> None:
        """Record a non-fatal warning."""

    def fail(self, stage: StageId, message: str) -> None:
        """Halt a feature (red exclamation) after an unrecoverable service error."""


class StageImplementation:
    """
    Base class for a stage implementation.

    Override `process` for per-photo work. Override `start`/`stop` for features that
    run a service while enabled (the FTP server); they are called when the user turns
    the feature on or off. Unregistered stages use this base, which does nothing.
    """

    def process(self, job: StageJob) -> StageResult:
        """
        Process one photo on a background thread.

        Raise any exception to fail (retried with backoff, then the feature halts) or
        `PermanentStageError` to halt immediately. Do not touch the UI.

        Returns:
            The result describing the files written.
        """
        return StageResult.skipped("not implemented")

    def start(self, host: StageHost) -> None:
        """Start the feature's service when it is switched on."""

    def stop(self) -> None:
        """Stop the feature's service when it is switched off."""

    def status_text(self) -> str:
        """
        Describe the running service for the status panel (shown while the feature is on).

        Returns:
            A short line such as connection details, or an empty string for none.
        """
        return ""


class StageRegistry:
    """Maps stages to their implementations."""

    def __init__(self) -> None:
        """Create a registry where every stage does nothing."""
        self._implementations: dict[StageId, StageImplementation] = {}

    def register(self, stage: StageId, implementation: StageImplementation) -> None:
        """
        Register the implementation of a stage.

        Raises:
            ValueError: If the stage already has an implementation.
        """
        if stage in self._implementations:
            raise ValueError(f"Stage {stage.value} already has an implementation")
        self._implementations[stage] = implementation

    def get(self, stage: StageId) -> StageImplementation:
        """Return a stage's implementation, or a no-op when none is registered."""
        return self._implementations.get(stage) or _NO_OP

    def is_registered(self, stage: StageId) -> bool:
        """Return whether a feature session has registered the stage."""
        return stage in self._implementations


_NO_OP = StageImplementation()
