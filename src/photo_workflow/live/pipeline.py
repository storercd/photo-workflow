"""Staged background pipeline: statuses, eligibility, invalidation, failures, warnings."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from photo_workflow.live.api import (
    LiveWarning,
    PermanentStageError,
    StageJob,
    StageRegistry,
    StageResult,
)
from photo_workflow.live.ingest import rename_arrival
from photo_workflow.live.layout import IMAGE_SUFFIXES, RAW_SUFFIXES, LiveLayout
from photo_workflow.live.persistence import (
    dump_state,
    load_state,
    restorable_records,
    write_state,
)
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import (
    SPECS,
    STAGE_ORDER,
    StageId,
    StageStatus,
    downstream_of,
    upstream_of,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_BACKOFF = (0.5, 1.5, 4.0)
CLEAN_DRAIN_TIMEOUT_SECONDS = 60.0
# Stages whose output can stand in for the original photo when viewing and cropping.
VIEW_STAGES = tuple(
    stage for stage in STAGE_ORDER if stage not in {StageId.FTP, *downstream_of(StageId.LIGHTING)}
)


@dataclass
class ImageStage:
    """Mutable state of one image in one stage."""

    status: StageStatus = StageStatus.PENDING
    outputs: tuple[Path, ...] = ()
    message: str = ""
    error: str = ""
    generation: int = 0


@dataclass(frozen=True)
class StageSummary:
    """Status-panel numbers for one stage."""

    stage: StageId
    count: int
    running: int
    failed: int
    enabled: bool
    waiting_disabled: bool
    halted: str = ""


@dataclass(frozen=True)
class PipelineSnapshot:
    """Status-panel numbers for the whole pipeline."""

    stages: tuple[StageSummary, ...]
    completed: int = 0
    held: bool = False


@dataclass(frozen=True)
class ViewSource:
    """The best available image for viewing a photo."""

    path: Path
    stage: StageId | None = None


@dataclass
class _Host:
    """`StageHost` handed to stage `start` hooks."""

    pipeline: LivePipeline
    stage: StageId

    @property
    def layout(self) -> LiveLayout:
        return self.pipeline.layout

    @property
    def settings(self) -> LiveSettings:
        return self.pipeline.settings

    def ingest(self, path: Path) -> None:
        self.pipeline.ingest(path)

    def warn(self, stage: StageId, message: str, photo: Path | None = None) -> None:
        self.pipeline.add_warning(stage, message, photo)

    def fail(self, stage: StageId, message: str) -> None:
        self.pipeline.halt(stage, message)


@dataclass
class _State:
    images: dict[Path, dict[StageId, ImageStage]] = field(default_factory=dict)
    halted: dict[StageId, str] = field(default_factory=dict)
    running: dict[StageId, int] = field(default_factory=dict)
    warnings: list[LiveWarning] = field(default_factory=list)
    incoming_seen: dict[Path, tuple[int, float]] = field(default_factory=dict)
    ingesting: set[Path] = field(default_factory=set)


class LivePipeline:
    """
    Per-image staged pipeline that runs on background threads.

    All public methods are thread-safe and never block on stage work. Listeners
    registered with `add_listener` are called (from any thread) after state changes.
    """

    def __init__(
        self,
        layout: LiveLayout,
        settings: LiveSettings,
        registry: StageRegistry,
        *,
        executor: Executor | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
    ) -> None:
        """Create a pipeline over the given layout; call `scan` to pick up photos."""
        self.layout = layout
        self.settings = settings
        self._registry = registry
        self._executor = executor or ThreadPoolExecutor(
            max_workers=len(STAGE_ORDER) + 1, thread_name_prefix="live"
        )
        self._sleep = sleep
        self._backoff = backoff
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._state = _State(running=dict.fromkeys(STAGE_ORDER, 0))
        self._listeners: list[Callable[[], None]] = []
        self._warning_ids = count(1)
        self._paused = False
        self._held = False
        self._closed = False
        self._saved_state: dict | None = None
        self._written_state = ""

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Call `listener` after every state change."""
        self._listeners.append(listener)

    def _emit(self) -> None:
        for listener in list(self._listeners):
            listener()

    # ----- photos -------------------------------------------------------------------

    def scan(self) -> None:
        """Register photos already in the Live folder and forget vanished ones."""
        found = set(self.layout.list_photos())
        with self._lock:
            if self._saved_state is None:
                self._saved_state = load_state(self.layout.root)
            for photo in list(self._state.images):
                if photo not in found:
                    del self._state.images[photo]
            for photo in sorted(found - self._state.images.keys()):
                self._register(photo)
            self._saved_state = {}  # progress is restored only for the photos present at launch
        self.pump()

    def _register(self, photo: Path) -> None:
        stages = {stage: ImageStage() for stage in STAGE_ORDER}
        stages[StageId.FTP].status = StageStatus.DONE
        restored = restorable_records(
            self.layout.root,
            self._saved_state or {},
            photo,
            self._enabled_flags(),
            self._autocrop_flags(),
        )
        for stage, (status, outputs, message) in restored.items():
            stages[stage].status = status
            stages[stage].outputs = outputs
            stages[stage].message = message
        self._state.images[photo] = stages

    def _enabled_flags(self) -> dict[str, bool]:
        return {stage.value: self.settings.is_enabled(stage) for stage in STAGE_ORDER}

    def _autocrop_flags(self) -> list:
        ratios, ball, force = _autocrop_options(self.settings)
        return [list(ratios), ball, force]

    def _save_state(self) -> None:
        """Write finished progress next to the photos so a restart resumes where it left off."""
        with self._lock:
            if not self._state.images or not self.layout.root.is_dir() or self._saved_state is None:
                return
            text = dump_state(
                self.layout.root,
                self._state.images,
                self._enabled_flags(),
                self._autocrop_flags(),
            )
            if text == self._written_state:
                return
            try:
                write_state(self.layout.root, text)
            except OSError:
                LOGGER.exception("could not save live pipeline state")
                return
            self._written_state = text

    def photos(self) -> list[Path]:
        """Return known photos in chronological order."""
        with self._lock:
            return sorted(self._state.images, key=lambda path: (path.name, str(path)))

    def ingest(self, path: Path) -> None:
        """Rename a completed upload on a background thread, then start its pipeline."""
        with self._lock:
            if path in self._state.ingesting or self._closed:
                return
            self._state.ingesting.add(path)
        self._executor.submit(self._ingest_one, path)
        self._emit()

    def poll_incoming(self) -> None:
        """
        Ingest landing files whose size and time stopped changing since last poll.

        Also registers renamed photos copied straight into a date folder.
        """
        files = self.layout.list_incoming()
        ready: list[Path] = []
        with self._lock:
            halted = StageId.FTP in self._state.halted
            seen = self._state.incoming_seen
            current = {path: _signature(path) for path in files}
            for path, signature in current.items():
                if not halted and seen.get(path) == signature:
                    ready.append(path)
            self._state.incoming_seen = {
                path: signature for path, signature in current.items() if path not in ready
            }
        for path in ready:
            self.ingest(path)
        if not self._paused:
            self.scan()
        self._emit()

    def _ingest_one(self, path: Path) -> None:
        try:
            target = self._with_retries(lambda: rename_arrival(self.layout, path))
        except Exception as error:
            LOGGER.exception("could not rename arrival %s", path)
            self.halt(StageId.FTP, f"Could not rename {path.name}: {error}")
            target = None
        with self._lock:
            self._state.ingesting.discard(path)
            if target is not None and not self._paused:
                if target in self._state.images:
                    self._invalidate(target, StageId.FTP, include_self=False)
                else:
                    self._register(target)
        self.pump()

    # ----- settings and toggles -----------------------------------------------------

    def toggle(self, stage: StageId, enabled: bool) -> None:
        """Switch a feature on or off (AutoCrop cannot be switched)."""
        self.set_settings(self.settings.with_enabled(stage, enabled))

    def set_settings(self, settings: LiveSettings) -> None:
        """
        Apply new settings.

        Turning a feature off clears its error; turning it on retries the images that
        were failing. Either change invalidates everything downstream of a stage that
        feeds later stages, because their source changed.
        """
        with self._lock:
            previous = self.settings
            self.settings = settings
            changed = [
                stage
                for stage in STAGE_ORDER
                if SPECS[stage].toggleable
                and previous.is_enabled(stage) != settings.is_enabled(stage)
            ]
            for stage in changed:
                self._apply_toggle(stage, settings.is_enabled(stage))
            if _autocrop_options(previous) != _autocrop_options(settings):
                for photo in self._state.images:
                    self._invalidate(photo, StageId.AUTOCROP, include_self=True)
        self._call_hooks(changed)
        self.pump()

    def _apply_toggle(self, stage: StageId, enabled: bool) -> None:
        if not enabled:
            self._state.halted.pop(stage, None)
        for photo, stages in self._state.images.items():
            record = stages[stage]
            if not enabled and record.status in {StageStatus.FAILED, StageStatus.RUNNING}:
                self._reset(record)
            # FTP only brings in new files; toggling it changes no existing image's source.
            if stage is not StageId.FTP and not SPECS[stage].blocks_when_disabled:
                self._invalidate(photo, stage, include_self=False)

    def _call_hooks(self, changed: list[StageId]) -> None:
        for stage in changed:
            implementation = self._registry.get(stage)
            try:
                if self.settings.is_enabled(stage):
                    implementation.start(_Host(self, stage))
                else:
                    implementation.stop()
            except Exception as error:
                LOGGER.exception("live %s service failed", stage.value)
                self.halt(stage, str(error))

    def start_enabled_features(self) -> None:
        """Start the services of features that are already enabled (call once at launch)."""
        self._call_hooks([stage for stage in STAGE_ORDER if self.settings.is_enabled(stage)])

    # ----- invalidation -------------------------------------------------------------

    def invalidate(
        self, stage: StageId, photo: Path | None = None, *, include_self: bool = False
    ) -> None:
        """
        Regenerate downstream work of a stage for one photo or all photos.

        Everything after `stage` goes back to pending and re-runs immediately; with
        `include_self` the stage itself is redone too.
        """
        with self._lock:
            targets = [photo] if photo is not None else list(self._state.images)
            for target in targets:
                if target in self._state.images:
                    self._invalidate(target, stage, include_self=include_self)
        self.pump()

    def _invalidate(self, photo: Path, stage: StageId, *, include_self: bool) -> None:
        stages = self._state.images[photo]
        affected = downstream_of(stage)
        if include_self:
            affected = (stage, *affected)
        for target in affected:
            record = stages[target]
            if record.status is not StageStatus.PENDING:
                self._reset(record)

    @staticmethod
    def _reset(record: ImageStage) -> None:
        record.status = StageStatus.PENDING
        record.outputs = ()
        record.message = ""
        record.error = ""
        record.generation += 1

    # ----- scheduling ---------------------------------------------------------------

    @property
    def held(self) -> bool:
        """Whether starting new stage work is on hold (running jobs may still finish)."""
        return self._held

    def set_held(self, held: bool) -> None:
        """Hold back new stage work, or release it and start whatever became waiting."""
        with self._lock:
            if self._held == held:
                return
            self._held = held
        LOGGER.info("live processing %s", "held" if held else "released")
        self.pump()

    def pump(self) -> None:
        """Start every stage that has eligible images and a free worker slot."""
        jobs: list[tuple[StageJob, int]] = []
        with self._lock:
            if self._paused or self._closed:
                return
            for stage in STAGE_ORDER:
                if self._held:
                    break
                jobs.extend(self._claim(stage))
        for job, generation in jobs:
            self._executor.submit(self._run_job, job, generation)
        self._save_state()
        self._emit()

    def _claim(self, stage: StageId) -> list[tuple[StageJob, int]]:
        spec = SPECS[stage]
        if stage in self._state.halted or not self.settings.is_enabled(stage):
            return []
        claimed = []
        for photo in self.photos():
            if self._state.running[stage] >= spec.concurrency:
                break
            if not self._eligible(photo, stage):
                continue
            record = self._state.images[photo][stage]
            record.status = StageStatus.RUNNING
            self._state.running[stage] += 1
            claimed.append((self._build_job(photo, stage), record.generation))
        return claimed

    def _eligible(self, photo: Path, stage: StageId) -> bool:
        if self._state.images[photo][stage].status is not StageStatus.PENDING:
            return False
        return all(self._settled(photo, upstream) for upstream in upstream_of(stage))

    def _settled(self, photo: Path, stage: StageId) -> bool:
        record = self._state.images[photo][stage]
        if record.status in {StageStatus.DONE, StageStatus.SKIPPED}:
            return True
        return not self.settings.is_enabled(stage) and not SPECS[stage].blocks_when_disabled

    def _build_job(self, photo: Path, stage: StageId) -> StageJob:
        source, source_stage = self._source(photo, stage)
        return StageJob(
            stage=stage,
            photo=photo,
            source=source,
            source_stage=source_stage,
            layout=self.layout,
            settings=self.settings,
            warn=lambda message: self.add_warning(stage, message, photo),
        )

    def _source(self, photo: Path, stage: StageId) -> tuple[Path, StageId | None]:
        for upstream in upstream_of(stage):
            if not self.settings.is_enabled(upstream):
                continue
            record = self._state.images[photo][upstream]
            if record.status is StageStatus.DONE and record.outputs and record.outputs[-1].exists():
                return record.outputs[-1], upstream
        return photo, None

    def view_source(self, photo: Path) -> ViewSource:
        """
        Return the best image to show for a photo.

        This is the most advanced finished output of a viewable stage whose file can
        be decoded, falling back through earlier stages to the original input.

        Returns:
            The file to show and the stage that produced it (`None` for the original).
        """
        with self._lock:
            stages = self._state.images.get(photo)
            for stage in reversed(VIEW_STAGES):
                if stages is None or not self.settings.is_enabled(stage):
                    continue
                record = stages[stage]
                output = record.outputs[-1] if record.outputs else None
                viewable = {*IMAGE_SUFFIXES, *RAW_SUFFIXES}
                if (
                    record.status is StageStatus.DONE
                    and output is not None
                    and output.suffix.lower() in viewable
                    and output.exists()
                ):
                    return ViewSource(output, stage)
        return ViewSource(photo)

    def stage_info(self) -> dict[StageId, str]:
        """
        Return the status lines of running feature services (for example the FTP address).

        Returns:
            Non-empty `status_text` of every enabled stage, keyed by stage.
        """
        info = {}
        for stage in STAGE_ORDER:
            if self.settings.is_enabled(stage) and stage not in self._state.halted:
                text = self._registry.get(stage).status_text()
                if text:
                    info[stage] = text
        return info

    # ----- running ------------------------------------------------------------------

    def _run_job(self, job: StageJob, generation: int) -> None:
        implementation = self._registry.get(job.stage)
        try:
            result = self._with_retries(lambda: implementation.process(job), job.stage)
        except Exception as error:
            self._finish(job, generation, None, f"{type(error).__name__}: {error}")
        else:
            self._finish(job, generation, result, "")

    def _with_retries[T](self, work: Callable[[], T], stage: StageId | None = None) -> T:
        attempts = len(self._backoff) + 1
        for attempt in range(attempts):
            try:
                return work()
            except PermanentStageError:
                raise
            except Exception as error:
                if attempt == attempts - 1:
                    raise
                delay = self._backoff[attempt]
                LOGGER.warning(
                    "live %s attempt %s/%s failed (%s); retrying in %.1fs",
                    stage.value if stage else "ingest",
                    attempt + 1,
                    attempts,
                    error,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _finish(
        self, job: StageJob, generation: int, result: StageResult | None, error: str
    ) -> None:
        with self._lock:
            self._state.running[job.stage] -= 1
            record = self._state.images.get(job.photo, {}).get(job.stage)
            if record is not None and record.generation == generation:
                if result is None:
                    self._record_failure(job, record, error)
                else:
                    self._record_success(job, record, result)
            self._idle.notify_all()
        self.pump()

    def _record_success(self, job: StageJob, record: ImageStage, result: StageResult) -> None:
        record.status = result.status
        record.outputs = result.outputs
        record.message = result.message
        record.error = ""
        LOGGER.info("live %s %s: %s", job.stage.value, job.photo.name, result.status.value)
        self._invalidate(job.photo, job.stage, include_self=False)

    def _record_failure(self, job: StageJob, record: ImageStage, error: str) -> None:
        record.status = StageStatus.FAILED
        record.error = error
        LOGGER.error("live %s failed for %s: %s", job.stage.value, job.photo.name, error)
        self._state.halted[job.stage] = error

    # ----- failure, warnings, status ------------------------------------------------

    def halt(self, stage: StageId, message: str) -> None:
        """Halt a feature and flag it as damaged until it is toggled off."""
        with self._lock:
            self._state.halted[stage] = message
        LOGGER.error("live %s halted: %s", stage.value, message)
        self._emit()

    def halted(self, stage: StageId) -> str:
        """Return the error that halted a feature, or an empty string."""
        with self._lock:
            return self._state.halted.get(stage, "")

    def failure(self, photo: Path) -> str:
        """Return the error of the first failed stage of a photo, or an empty string."""
        with self._lock:
            for record in self._state.images.get(photo, {}).values():
                if record.status is StageStatus.FAILED:
                    return record.error
        return ""

    def skip_message(self, photo: Path, stage: StageId) -> str:
        """Return why a photo's stage was skipped, or an empty string when it was not."""
        with self._lock:
            record = self._state.images.get(photo, {}).get(stage)
            if record is None or record.status is not StageStatus.SKIPPED:
                return ""
            return record.message

    def status(self, photo: Path, stage: StageId) -> StageStatus:
        """Return one image's status in one stage."""
        with self._lock:
            return self._state.images[photo][stage].status

    def add_warning(self, stage: StageId, message: str, photo: Path | None = None) -> None:
        """Record a non-fatal warning."""
        with self._lock:
            warning = LiveWarning(next(self._warning_ids), stage, message, photo)
            self._state.warnings.append(warning)
        LOGGER.warning("live %s warning: %s", stage.value, message)
        self._emit()

    def warnings(self) -> list[LiveWarning]:
        """Return the current warnings, oldest first."""
        with self._lock:
            return list(self._state.warnings)

    def clear_warnings(self, warning_id: int | None = None) -> None:
        """Dismiss one warning, or all of them."""
        with self._lock:
            self._state.warnings = [
                warning
                for warning in self._state.warnings
                if warning_id is not None and warning.id != warning_id
            ]
        self._emit()

    def snapshot(self) -> PipelineSnapshot:
        """
        Count images per stage for the status panel.

        An image counts at the first stage it has not passed; a blocking stage that is
        off (Export, Cloud Sync) holds images there. A feature that is off but passes
        images through (Denoise, Mask, Lighting/Color) also counts every image that has
        reached it, in addition to the stage it is really at, so the user can see the work
        waiting for that feature. Off features are flagged `waiting_disabled` so the
        panel can blink their counts.

        Returns:
            Per-stage counts plus the number of images that finished every stage.
        """
        with self._lock:
            counts = dict.fromkeys(STAGE_ORDER, 0)
            failed = dict.fromkeys(STAGE_ORDER, 0)
            completed = 0
            for photo, stages in self._state.images.items():
                position = next(
                    (stage for stage in STAGE_ORDER if not self._settled(photo, stage)), None
                )
                if position is None:
                    completed += 1
                    continue
                counts[position] += 1
                failed[position] += stages[position].status is StageStatus.FAILED
                for stage in STAGE_ORDER:
                    if self._is_waiting_on_disabled(photo, stage):
                        counts[stage] += 1
            counts[StageId.FTP] += len(self._state.ingesting) + len(self._state.incoming_seen)
            summaries = tuple(
                self._summarize(stage, counts[stage], failed[stage]) for stage in STAGE_ORDER
            )
        return PipelineSnapshot(summaries, completed, self._held)

    def _is_waiting_on_disabled(self, photo: Path, stage: StageId) -> bool:
        spec = SPECS[stage]
        return (
            spec.toggleable
            and not spec.blocks_when_disabled
            and stage is not StageId.FTP
            and not self.settings.is_enabled(stage)
            and self._eligible(photo, stage)
        )

    def _summarize(self, stage: StageId, total: int, failed: int) -> StageSummary:
        enabled = self.settings.is_enabled(stage)
        return StageSummary(
            stage=stage,
            count=total,
            running=self._state.running[stage],
            failed=failed,
            enabled=enabled,
            waiting_disabled=bool(total) and not enabled and stage is not StageId.FTP,
            halted=self._state.halted.get(stage, ""),
        )

    # ----- lifecycle ----------------------------------------------------------------

    def clean(self) -> None:
        """
        Wipe the whole Live folder, including the input originals.

        Waits for running jobs to finish first so nothing recreates deleted folders,
        so call it off the UI thread.
        """
        with self._idle:
            self._paused = True
            self._idle.wait_for(
                lambda: not any(self._state.running.values()), CLEAN_DRAIN_TIMEOUT_SECONDS
            )
            self._state = _State(running=dict.fromkeys(STAGE_ORDER, 0))
            self._saved_state = {}
            self._written_state = ""
            self.layout.clean()
            self._paused = False
        LOGGER.info("cleaned live folder %s", self.layout.root)
        self.pump()

    def shutdown(self) -> None:
        """Stop starting work, stop feature services, and release the worker threads."""
        with self._lock:
            self._closed = True
            enabled = [stage for stage in STAGE_ORDER if self.settings.is_enabled(stage)]
        for stage in enabled:
            try:
                self._registry.get(stage).stop()
            except Exception:
                LOGGER.exception("live %s service failed to stop", stage.value)
        self._executor.shutdown(wait=False, cancel_futures=True)


def _signature(path: Path) -> tuple[int, float]:
    try:
        stat = path.stat()
    except OSError:
        return (-1, 0.0)
    return (stat.st_size, stat.st_mtime)


def _autocrop_options(settings: LiveSettings) -> tuple[tuple[str, ...], bool, bool]:
    return (settings.autocrop_ratios, settings.autocrop_ball, settings.autocrop_force)
