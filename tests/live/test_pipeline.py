"""Tests for the Live pipeline framework."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import replace
from pathlib import Path

import pytest

from photo_workflow.live.api import (
    PermanentStageError,
    StageImplementation,
    StageJob,
    StageRegistry,
    StageResult,
)
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.pipeline import LivePipeline
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import SPECS, StageId, StageStatus

PHOTO_NAME = "20251031_101500_00_IMG_0001.CR3"


class ManualExecutor(Executor):
    """Queues work so tests decide when background jobs run."""

    def __init__(self) -> None:
        """Start with an empty queue."""
        self.queue: list[Callable[[], None]] = []

    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> Future:
        """
        Queue a call instead of running it.

        Returns:
            A future that is never resolved by the executor itself.
        """
        self.queue.append(lambda: fn(*args, **kwargs))
        return Future()

    def run_all(self) -> None:
        """Run queued work, including work queued by that work, until idle."""
        while self.queue:
            self.queue.pop(0)()


class Recorder(StageImplementation):
    """Stage that writes one file and records what it was given."""

    def __init__(self, subfolder_suffix: str = ".out") -> None:
        """Create a recorder writing files with the given suffix."""
        self.calls: list[StageJob] = []
        self.suffix = subfolder_suffix

    def process(self, job: StageJob) -> StageResult:
        """
        Write an output file and remember the job.

        Returns:
            A done result listing the file.
        """
        self.calls.append(job)
        target = job.output_path(suffix=self.suffix)
        target.write_text(job.source.name)
        return StageResult.done(target)


class Flaky(StageImplementation):
    """Stage that raises for a number of calls before succeeding."""

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        """Fail `failures` times (use a large number to always fail)."""
        self.failures = failures
        self.error = error or RuntimeError("network down")
        self.calls = 0

    def process(self, job: StageJob) -> StageResult:
        """
        Raise until the failure budget is used up.

        Returns:
            A skipped result once the budget is exhausted.
        """
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error  # noqa: DOC501
        return StageResult.skipped("recovered")


class Calls(StageImplementation):
    """Stage that does no file work and only counts how often it ran."""

    def __init__(self) -> None:
        """Create the counter."""
        self.calls: list[StageJob] = []

    def process(self, job: StageJob) -> StageResult:
        """
        Remember the job.

        Returns:
            A done result with no outputs.
        """
        self.calls.append(job)
        return StageResult.done()


def make_photo(root: Path, name: str = PHOTO_NAME) -> Path:
    """
    Create a renamed original in a date folder.

    Returns:
        The photo path.
    """
    input_dir = root / "2025" / "10" / "20251031" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    photo = input_dir / name
    photo.write_bytes(b"raw")
    return photo


def make_pipeline(
    tmp_path: Path,
    registry: StageRegistry | None = None,
    enabled: set[StageId] | None = None,
) -> tuple[LivePipeline, ManualExecutor, list[float]]:
    """
    Create a pipeline over `tmp_path` with a manual executor and recorded sleeps.

    Returns:
        The pipeline, its executor and the list of backoff delays it slept.
    """
    executor = ManualExecutor()
    sleeps: list[float] = []
    settings = LiveSettings(enabled=frozenset(enabled or ()))
    pipeline = LivePipeline(
        LiveLayout(tmp_path),
        settings,
        registry or StageRegistry(),
        executor=executor,
        sleep=sleeps.append,
        backoff=(0.1, 0.2),
    )
    return pipeline, executor, sleeps


ALL_FEATURES = {
    StageId.FTP,
    StageId.DENOISE,
    StageId.MASK,
    StageId.LIGHTING,
    StageId.EXPORT,
    StageId.CLOUD_SYNC,
}


def test_features_are_off_by_default_and_images_wait_at_blocking_export(tmp_path: Path) -> None:
    """Soft stages pass through but show waiting work; Export holds images."""
    photo = make_photo(tmp_path)
    pipeline, executor, _ = make_pipeline(tmp_path)

    pipeline.scan()
    executor.run_all()

    assert pipeline.status(photo, StageId.AUTOCROP) is StageStatus.SKIPPED
    summaries = {summary.stage: summary for summary in pipeline.snapshot().stages}
    assert summaries[StageId.EXPORT].count == 1
    assert summaries[StageId.EXPORT].waiting_disabled
    assert summaries[StageId.DENOISE].count == 1
    assert summaries[StageId.DENOISE].waiting_disabled
    assert summaries[StageId.CLOUD_SYNC].count == 0


def test_autocrop_can_be_disabled_and_reenabled(tmp_path: Path) -> None:
    """A disabled AutoCrop does no work and lets Export through; re-enabling runs it."""
    make_photo(tmp_path)
    autocrop = Calls()
    registry = StageRegistry()
    registry.register(StageId.AUTOCROP, autocrop)
    pipeline, executor, _ = make_pipeline(tmp_path, registry)
    pipeline.toggle(StageId.AUTOCROP, False)

    pipeline.scan()
    executor.run_all()

    assert not pipeline.settings.is_enabled(StageId.AUTOCROP)
    assert autocrop.calls == []
    summary = {s.stage: s for s in pipeline.snapshot().stages}[StageId.AUTOCROP]
    assert summary.waiting_disabled

    pipeline.toggle(StageId.AUTOCROP, True)
    executor.run_all()

    assert len(autocrop.calls) == 1


def test_stage_uses_most_advanced_earlier_output_and_falls_back(tmp_path: Path) -> None:
    """Missing or disabled upstream outputs fall back as far as the original input."""
    photo = make_photo(tmp_path)
    denoise, lighting = Recorder(".dng"), Recorder(".jpg")
    registry = StageRegistry()
    registry.register(StageId.DENOISE, denoise)
    registry.register(StageId.LIGHTING, lighting)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.LIGHTING})

    pipeline.scan()
    executor.run_all()

    assert denoise.calls == []
    assert lighting.calls[0].source == photo
    assert lighting.calls[0].source_stage is None

    pipeline.toggle(StageId.DENOISE, True)
    executor.run_all()

    assert denoise.calls[0].source == photo
    assert lighting.calls[-1].source.name == photo.with_suffix(".dng").name
    assert lighting.calls[-1].source_stage is StageId.DENOISE


def test_regenerating_a_stage_invalidates_and_reruns_downstream(tmp_path: Path) -> None:
    """Redoing an upstream stage re-runs everything after it immediately."""
    make_photo(tmp_path)
    mask, lighting = Recorder(".mask"), Recorder(".jpg")
    registry = StageRegistry()
    registry.register(StageId.MASK, mask)
    registry.register(StageId.LIGHTING, lighting)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.MASK, StageId.LIGHTING})
    pipeline.scan()
    executor.run_all()
    assert (len(mask.calls), len(lighting.calls)) == (1, 1)

    pipeline.invalidate(StageId.MASK, include_self=True)
    executor.run_all()

    assert (len(mask.calls), len(lighting.calls)) == (2, 2)
    assert lighting.calls[-1].source_stage is StageId.MASK


def test_prioritized_photo_is_regenerated_first(tmp_path: Path) -> None:
    """After an invalidation the photo being viewed is served before the others."""
    make_photo(tmp_path, "20251031_101500_00_IMG_0001.CR3")
    viewed = make_photo(tmp_path, "20251031_101500_00_IMG_0002.CR3")
    mask = Recorder(".mask")
    registry = StageRegistry()
    registry.register(StageId.MASK, mask)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.MASK})
    pipeline.scan()
    executor.run_all()
    mask.calls.clear()

    pipeline.prioritize(viewed)
    pipeline.invalidate(StageId.MASK, include_self=True)
    executor.run_all()

    assert [job.photo for job in mask.calls] == [viewed, mask.calls[1].photo]
    assert mask.calls[1].photo != viewed


def test_result_of_invalidated_job_is_discarded(tmp_path: Path) -> None:
    """An in-flight job whose input was invalidated does not record a stale result."""
    photo = make_photo(tmp_path)
    mask = Recorder(".mask")
    registry = StageRegistry()
    registry.register(StageId.MASK, mask)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.MASK})
    pipeline.scan()
    assert pipeline.status(photo, StageId.MASK) is StageStatus.RUNNING

    pipeline.invalidate(StageId.MASK, include_self=True)
    executor.run_all()

    assert len(mask.calls) == 2
    assert pipeline.status(photo, StageId.MASK) is StageStatus.DONE


def test_progress_names_the_stage_working_on_a_photo(tmp_path: Path) -> None:
    """A photo reports its running stage, then nothing once every enabled stage is done."""
    photo = make_photo(tmp_path)
    registry = StageRegistry()
    registry.register(StageId.MASK, Recorder(".mask"))
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.MASK})
    pipeline.scan()

    assert pipeline.progress(photo) == f"Processing: {SPECS[StageId.MASK].label}"

    executor.run_all()

    assert pipeline.progress(photo) == ""


def test_failure_retries_then_halts_feature_and_toggle_clears_and_retries(
    tmp_path: Path,
) -> None:
    """Retries use backoff, exhaustion halts the feature, off/on recovers failed images."""
    first = make_photo(tmp_path)
    second = make_photo(tmp_path, "20251031_101501_00_IMG_0002.CR3")
    flaky = Flaky(failures=3)
    registry = StageRegistry()
    registry.register(StageId.DENOISE, flaky)
    pipeline, executor, sleeps = make_pipeline(tmp_path, registry, {StageId.DENOISE})

    pipeline.scan()
    executor.run_all()

    assert sleeps == [0.1, 0.2]
    assert flaky.calls == 3
    assert pipeline.status(first, StageId.DENOISE) is StageStatus.FAILED
    assert "network down" in pipeline.failure(first)
    assert "network down" in pipeline.halted(StageId.DENOISE)
    assert pipeline.status(second, StageId.DENOISE) is StageStatus.PENDING

    pipeline.toggle(StageId.DENOISE, False)
    assert pipeline.halted(StageId.DENOISE) == ""
    assert pipeline.failure(first) == ""
    executor.run_all()

    pipeline.toggle(StageId.DENOISE, True)
    executor.run_all()

    assert pipeline.status(first, StageId.DENOISE) is StageStatus.SKIPPED
    assert pipeline.status(second, StageId.DENOISE) is StageStatus.SKIPPED
    assert pipeline.failure(first) == ""


def test_permanent_error_halts_without_retries(tmp_path: Path) -> None:
    """A permanent error skips the backoff retries."""
    photo = make_photo(tmp_path)
    flaky = Flaky(failures=99, error=PermanentStageError("corrupt file"))
    registry = StageRegistry()
    registry.register(StageId.MASK, flaky)
    pipeline, executor, sleeps = make_pipeline(tmp_path, registry, {StageId.MASK})

    pipeline.scan()
    executor.run_all()

    assert flaky.calls == 1
    assert sleeps == []
    assert pipeline.status(photo, StageId.MASK) is StageStatus.FAILED
    assert pipeline.halted(StageId.MASK)


def test_failed_stage_holds_image_in_status_snapshot(tmp_path: Path) -> None:
    """A failed image stays counted at the failed stage."""
    make_photo(tmp_path)
    registry = StageRegistry()
    registry.register(StageId.LIGHTING, Flaky(failures=99))
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.LIGHTING})

    pipeline.scan()
    executor.run_all()

    summary = {s.stage: s for s in pipeline.snapshot().stages}[StageId.LIGHTING]
    assert (summary.count, summary.failed) == (1, 1)
    assert summary.halted


def test_enabling_export_releases_waiting_images(tmp_path: Path) -> None:
    """Turning Export on lets waiting images move on to Cloud Sync."""
    photo = make_photo(tmp_path)
    pipeline, executor, _ = make_pipeline(tmp_path)
    pipeline.scan()
    executor.run_all()

    pipeline.toggle(StageId.EXPORT, True)
    executor.run_all()

    assert pipeline.status(photo, StageId.EXPORT) is StageStatus.SKIPPED
    summaries = {s.stage: s for s in pipeline.snapshot().stages}
    assert summaries[StageId.CLOUD_SYNC].count == 1
    assert summaries[StageId.CLOUD_SYNC].waiting_disabled


def test_view_source_prefers_final_output_then_falls_back(tmp_path: Path) -> None:
    """The viewer gets the output JPEG when it exists, else the original."""
    photo = make_photo(tmp_path)
    registry = StageRegistry()
    registry.register(StageId.LIGHTING, Recorder(".jpg"))
    pipeline, executor, _ = make_pipeline(tmp_path, registry)
    pipeline.scan()
    executor.run_all()
    assert pipeline.view_source(photo).path == photo

    pipeline.toggle(StageId.LIGHTING, True)
    executor.run_all()

    source = pipeline.view_source(photo)
    assert source.stage is StageId.LIGHTING
    assert source.path.parent.name == "lighting"


def test_warnings_can_be_listed_and_dismissed(tmp_path: Path) -> None:
    """Warnings are recorded without failing anything."""
    photo = make_photo(tmp_path)
    pipeline, _, _ = make_pipeline(tmp_path)

    pipeline.add_warning(StageId.FTP, "upload interrupted", photo)
    warning = pipeline.warnings()[0]
    pipeline.clear_warnings(warning.id)

    assert warning.message == "upload interrupted"
    assert pipeline.warnings() == []
    assert pipeline.halted(StageId.FTP) == ""


def test_clean_wipes_live_folder_including_input(tmp_path: Path) -> None:
    """Clean removes originals and forgets every image."""
    photo = make_photo(tmp_path)
    pipeline, executor, _ = make_pipeline(tmp_path)
    pipeline.scan()
    executor.run_all()

    pipeline.clean()

    assert not photo.exists()
    assert pipeline.photos() == []
    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []


def test_arrival_is_renamed_before_entering_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newly arrived files are renamed with the card-import naming, overwriting targets."""
    from datetime import datetime  # noqa: PLC0415

    from photo_workflow.live import ingest  # noqa: PLC0415

    layout = LiveLayout(tmp_path)
    layout.incoming.mkdir()
    arrived = layout.incoming / "IMG_0001.CR3"
    arrived.write_bytes(b"new")
    monkeypatch.setattr(
        ingest, "read_arrival_timestamp", lambda path: datetime(2025, 10, 31, 10, 15, 0, 120000)
    )
    pipeline, executor, _ = make_pipeline(tmp_path)

    existing = tmp_path / "2025" / "10" / "20251031" / "input" / "20251031_101500_12_IMG_0001.CR3"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    pipeline.ingest(arrived)
    executor.run_all()

    assert not arrived.exists()
    assert existing.read_bytes() == b"new"
    assert pipeline.photos() == [existing]


def test_files_copied_into_the_live_folder_are_ingested_and_ready_for_denoise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAW files dropped in the Live root (or a date folder) are picked up without a restart."""
    from datetime import datetime  # noqa: PLC0415

    from photo_workflow.live import ingest  # noqa: PLC0415

    monkeypatch.setattr(
        ingest, "read_arrival_timestamp", lambda path: datetime(2025, 10, 31, 10, 15, 0, 0)
    )
    pipeline, executor, _ = make_pipeline(tmp_path)
    pipeline.scan()
    dropped = tmp_path / "IMG_0001.CR3"
    dropped.write_bytes(b"raw")
    already_named = make_photo(tmp_path, "20251101_090000_00_IMG_0002.CR3")

    pipeline.poll_incoming()
    pipeline.poll_incoming()
    executor.run_all()

    renamed = tmp_path / "2025" / "10" / "20251031" / "input" / "20251031_101500_00_IMG_0001.CR3"
    assert not dropped.exists()
    assert set(pipeline.photos()) == {renamed, already_named}
    assert pipeline.status(renamed, StageId.FTP) is StageStatus.DONE
    assert pipeline.snapshot().stages[0].count == 0


def test_already_stamped_arrivals_get_a_single_fresh_prefix(tmp_path: Path) -> None:
    """A file that already has capture-time prefixes is renamed with just the new one."""
    from datetime import datetime  # noqa: PLC0415

    from photo_workflow.live.ingest import rename_arrival  # noqa: PLC0415

    layout = LiveLayout(tmp_path)
    arrived = tmp_path / "20240101_080000_05_20240101_080000_05_IMG_0001.CR3"
    arrived.write_bytes(b"raw")

    target = rename_arrival(layout, arrived, captured_at=datetime(2025, 10, 31, 10, 15, 0))

    assert target.name == "20251031_101500_00_IMG_0001.CR3"
    assert target.parent == tmp_path / "2025" / "10" / "20251031" / "input"


def test_toggling_ftp_does_not_redo_existing_images(tmp_path: Path) -> None:
    """FTP only brings in new files, so switching it on or off leaves processed images alone."""
    make_photo(tmp_path)
    denoise = Recorder(".tif")
    registry = StageRegistry()
    registry.register(StageId.DENOISE, denoise)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.DENOISE})
    pipeline.scan()
    executor.run_all()
    assert len(denoise.calls) == 1

    pipeline.toggle(StageId.FTP, True)
    executor.run_all()
    pipeline.toggle(StageId.FTP, False)
    executor.run_all()

    assert len(denoise.calls) == 1


def test_restart_keeps_finished_work_and_redoes_what_changed(tmp_path: Path) -> None:
    """A new pipeline over the same folder resumes instead of redoing finished stages."""
    photo = make_photo(tmp_path)
    denoise = Recorder(".tif")
    registry = StageRegistry()
    registry.register(StageId.DENOISE, denoise)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.DENOISE})
    pipeline.scan()
    executor.run_all()
    assert len(denoise.calls) == 1

    again = Recorder(".tif")
    registry = StageRegistry()
    registry.register(StageId.DENOISE, again)
    restarted, executor, _ = make_pipeline(tmp_path, registry, {StageId.DENOISE})
    restarted.scan()
    executor.run_all()
    assert again.calls == []
    assert restarted.status(photo, StageId.DENOISE) is StageStatus.DONE

    # A missing output, a changed original, or a toggled feature means the work is redone.
    denoise.calls[0].output_path(suffix=".tif").unlink()
    redo = Recorder(".tif")
    registry = StageRegistry()
    registry.register(StageId.DENOISE, redo)
    third, executor, _ = make_pipeline(tmp_path, registry, {StageId.DENOISE})
    third.scan()
    executor.run_all()
    assert len(redo.calls) == 1


def test_restart_redoes_a_stage_whose_implementation_revision_changed(tmp_path: Path) -> None:
    """Output from an older revision of a stage is regenerated, with its downstream stages."""
    make_photo(tmp_path)
    enabled = {StageId.DENOISE, StageId.LIGHTING}
    registry = StageRegistry()
    registry.register(StageId.DENOISE, Recorder(".tif"))
    registry.register(StageId.LIGHTING, Recorder(".jpg"))
    pipeline, executor, _ = make_pipeline(tmp_path, registry, enabled)
    pipeline.scan()
    executor.run_all()

    newer_denoise, lighting = Recorder(".tif"), Recorder(".jpg")
    newer_denoise.revision = 2
    registry = StageRegistry()
    registry.register(StageId.DENOISE, newer_denoise)
    registry.register(StageId.LIGHTING, lighting)
    restarted, executor, _ = make_pipeline(tmp_path, registry, enabled)
    restarted.scan()
    executor.run_all()

    assert len(newer_denoise.calls) == 1
    assert len(lighting.calls) == 1


def test_holding_blocks_new_work_but_not_ingest_and_release_resumes(tmp_path: Path) -> None:
    """While held, images are still found and queued, but nothing starts until release."""
    denoise = Recorder(".tif")
    registry = StageRegistry()
    registry.register(StageId.DENOISE, denoise)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.DENOISE})
    pipeline.set_held(True)

    make_photo(tmp_path)
    pipeline.scan()
    executor.run_all()

    snapshot = pipeline.snapshot()
    assert snapshot.held
    assert not denoise.calls
    denoise_row = next(row for row in snapshot.stages if row.stage is StageId.DENOISE)
    assert denoise_row.count == 1
    assert not denoise_row.waiting_disabled

    pipeline.set_held(False)
    executor.run_all()

    assert len(denoise.calls) == 1
    assert not pipeline.snapshot().held


def test_changing_the_color_temperature_reruns_lighting(tmp_path: Path) -> None:
    """A new white balance reprocesses every photo from Lighting onward."""
    make_photo(tmp_path)
    lighting = Calls()
    registry = StageRegistry()
    registry.register(StageId.LIGHTING, lighting)
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.LIGHTING})
    pipeline.scan()
    executor.run_all()

    pipeline.set_settings(replace(pipeline.settings, color_temperature=4200.0))
    executor.run_all()

    assert len(lighting.calls) == 2


def test_invalidating_a_running_stage_cancels_it_at_its_next_progress_report(
    tmp_path: Path,
) -> None:
    """A stale job stops at its next progress report instead of finishing, then reruns."""
    photo = make_photo(tmp_path)
    attempts: list[str] = []

    class Slow(StageImplementation):
        def process(self, job: StageJob) -> StageResult:
            attempts.append("start")
            if len(attempts) == 1:
                pipeline.invalidate(StageId.LIGHTING, photo, include_self=True)
            job.report_progress(0.5)
            attempts.append("finish")
            return StageResult.done()

    registry = StageRegistry()
    registry.register(StageId.LIGHTING, Slow())
    pipeline, executor, _ = make_pipeline(tmp_path, registry, {StageId.LIGHTING})
    pipeline.scan()
    executor.run_all()

    assert attempts == ["start", "start", "finish"]
    assert pipeline.status(photo, StageId.LIGHTING) is StageStatus.DONE
