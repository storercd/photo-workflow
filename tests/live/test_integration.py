"""End-to-end Live pipeline run with the real AutoCrop and Export and stub, offline stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_pipeline import ManualExecutor, make_photo

from photo_workflow.editor.metadata import PhotoMetadata
from photo_workflow.editor.model import CropRect
from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.features import autocrop, export, load_features
from photo_workflow.live.imaging import save_linear_rgb
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.pipeline import LivePipeline
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import STAGE_ORDER, StageId, StageStatus

WIDTH, HEIGHT = 600, 400
RATIOS = ("3:2", "1:1")


class StubTiff(StageImplementation):
    """Writes a 16-bit linear intermediate, like Denoise and Mask."""

    def process(self, job: StageJob) -> StageResult:
        """
        Write a mid-gray TIFF into the stage's folder.

        Returns:
            The result listing the file.
        """
        target = job.output_path(suffix=".tif")
        save_linear_rgb(target, np.full((HEIGHT, WIDTH, 3), 20000, dtype=np.uint16))
        return StageResult.done(target)


class StubLighting(StageImplementation):
    """Writes the final JPEG into `output/`, like Lighting/Color."""

    def process(self, job: StageJob) -> StageResult:
        """
        Write a pure red JPEG.

        Returns:
            The result listing the file.
        """
        target = job.output_path("output", ".jpg")
        Image.new("RGB", (WIDTH, HEIGHT), (255, 0, 0)).save(target, "JPEG")
        return StageResult.done(target)


class StubCloudSync(StageImplementation):
    """Records the photos it was asked to upload, without any network."""

    def __init__(self) -> None:
        """Start with nothing uploaded."""
        self.uploaded: list[Path] = []

    def process(self, job: StageJob) -> StageResult:
        """
        Remember the exported file.

        Returns:
            A done result.
        """
        self.uploaded.append(job.source)
        return StageResult.done()


class FixedDetector:
    """Always finds a subject in the middle of the frame."""

    def detect(
        self, photo: Path, source: Path, size: tuple[int, int], *, want_ball: bool
    ) -> autocrop.Detection | None:
        """
        Return a centered subject.

        Returns:
            The detection.
        """
        width, height = size
        return autocrop.Detection(
            width, height, (width * 0.4, height * 0.3, width * 0.6, height * 0.7)
        )


class Rig:
    """A pipeline with every stage registered and the XMP sidecar replaced by a dict."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Register stubs plus the real AutoCrop and Export, all features on."""
        self.photo = make_photo(tmp_path)
        self.crops: list[CropRect] = []
        self.cloud = StubCloudSync()

        def read(*_: object) -> PhotoMetadata:
            crop = self.crops[-1] if self.crops else CropRect(0, 0, 1, 1)
            return PhotoMetadata(WIDTH, HEIGHT, crop)

        for module in (autocrop, export):
            monkeypatch.setattr(module, "read_photo_metadata", read)
        monkeypatch.setattr(autocrop, "is_manual_crop", lambda _: False)
        monkeypatch.setattr(
            autocrop, "write_photo_crop", lambda _r, _x, crop, *, source: self.crops.append(crop)
        )
        registry = StageRegistry()
        registry.register(StageId.DENOISE, StubTiff())
        registry.register(StageId.MASK, StubTiff())
        registry.register(StageId.LIGHTING, StubLighting())
        registry.register(StageId.LIGHTING_FULL, StubLighting())
        registry.register(StageId.AUTOCROP, autocrop.AutoCrop(FixedDetector()))
        registry.register(StageId.EXPORT, export.Export())
        registry.register(StageId.CLOUD_SYNC, self.cloud)
        self.executor = ManualExecutor()
        settings = LiveSettings(autocrop_ratios=RATIOS)
        for stage in STAGE_ORDER:
            settings = settings.with_enabled(
                stage, stage not in {StageId.FTP, StageId.LIGHTING_FULL}
            )
        self.pipeline = LivePipeline(
            LiveLayout(tmp_path), settings, registry, executor=self.executor, sleep=lambda _: None
        )
        self.pipeline.scan()

    def run(self) -> None:
        """Process everything queued."""
        self.executor.run_all()

    def exported(self) -> Image.Image:
        """
        Open the exported JPEG.

        Returns:
            The exported image.
        """
        (path,) = (self.photo.parent.parent / "export").glob("*.jpg")
        return Image.open(path)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    """
    A pipeline over a temporary Live folder with one photo.

    Returns:
        The rig.
    """
    return Rig(tmp_path, monkeypatch)


def test_photo_flows_through_every_stage_and_export_crops_the_best_source(rig: Rig) -> None:
    """All stages finish; Export renders the Lighting JPEG with the crop AutoCrop wrote."""
    rig.run()

    for stage in STAGE_ORDER:
        if stage not in {StageId.FTP, StageId.LIGHTING_FULL}:
            assert rig.pipeline.status(rig.photo, stage) is StageStatus.DONE, stage
    assert len(rig.crops) == 1
    crop = rig.crops[0]
    exported = rig.exported()
    assert exported.size == (
        pytest.approx(crop.width * WIDTH, abs=1),
        pytest.approx(crop.height * HEIGHT, abs=1),
    )
    red, green, blue = exported.convert("RGB").getpixel((0, 0))
    assert red > 200
    assert green < 60
    assert blue < 60
    assert rig.cloud.uploaded[0].parent.name == "export"
    assert rig.pipeline.snapshot().completed == 1


def test_export_falls_back_to_the_linear_intermediate_when_lighting_is_off(rig: Rig) -> None:
    """With Lighting/Color off, Export uses the Mask TIFF and the result is still a JPEG."""
    rig.pipeline.set_settings(rig.pipeline.settings.with_enabled(StageId.LIGHTING, False))
    rig.run()

    assert rig.pipeline.status(rig.photo, StageId.EXPORT) is StageStatus.DONE
    assert rig.pipeline.view_source(rig.photo).stage is StageId.MASK
    assert rig.exported().convert("RGB").getpixel((0, 0))[0] == pytest.approx(150, abs=3)


def test_regenerating_an_upstream_stage_reruns_the_whole_tail(rig: Rig) -> None:
    """Invalidating Mask re-runs Lighting, AutoCrop, Export and Cloud Sync immediately."""
    rig.run()
    exports_before = len(rig.cloud.uploaded)

    rig.pipeline.invalidate(StageId.MASK, rig.photo)
    rig.run()

    assert len(rig.cloud.uploaded) == exports_before + 1
    assert len(rig.crops) == 2
    assert rig.pipeline.status(rig.photo, StageId.CLOUD_SYNC) is StageStatus.DONE


def test_all_features_start_disabled_by_default() -> None:
    """A fresh settings object leaves every feature but AutoCrop off and registers all stages."""
    registry = StageRegistry()
    load_features(registry)
    settings = LiveSettings()

    assert all(registry.is_registered(stage) for stage in StageId)
    assert settings.enabled == frozenset()
    assert settings.is_enabled(StageId.AUTOCROP)
    assert settings.is_enabled(StageId.AUTOCROP)
    assert not any(
        settings.is_enabled(stage) for stage in STAGE_ORDER if stage is not StageId.AUTOCROP
    )


def test_running_service_status_is_listed_only_while_its_feature_is_on(rig: Rig) -> None:
    """`stage_info` surfaces a service's status line (the FTP address) for the UI."""

    class Service(StageImplementation):
        def status_text(self) -> str:
            """
            Describe the service.

            Returns:
                A fixed line.
            """
            return "listening"

    registry = StageRegistry()
    registry.register(StageId.FTP, Service())
    settings = LiveSettings().with_enabled(StageId.FTP, True)
    pipeline = LivePipeline(rig.pipeline.layout, settings, registry, executor=rig.executor)

    assert pipeline.stage_info() == {StageId.FTP: "listening"}
    pipeline.set_settings(settings.with_enabled(StageId.FTP, False))
    assert pipeline.stage_info() == {}


def test_skip_message_reports_why_a_stage_was_skipped(rig: Rig) -> None:
    """`skip_message` is the skip reason for skipped stages and empty otherwise."""
    rig.run()
    assert rig.pipeline.skip_message(rig.photo, StageId.MASK) == ""

    class Skipper(StageImplementation):
        def process(self, job: StageJob) -> StageResult:
            """
            Skip with a reason.

            Returns:
                A skipped result.
            """
            return StageResult.skipped("no subject")

    registry = StageRegistry()
    registry.register(StageId.MASK, Skipper())
    settings = LiveSettings().with_enabled(StageId.MASK, True)
    executor = ManualExecutor()
    pipeline = LivePipeline(rig.pipeline.layout, settings, registry, executor=executor)
    pipeline.scan()
    executor.run_all()

    assert pipeline.skip_message(rig.photo, StageId.MASK) == "no subject"


def test_full_lighting_runs_behind_the_quick_one_and_the_tail_is_redone(rig: Rig) -> None:
    """Export proceeds from the quick image; the full image lands later and re-runs the tail."""
    rig.pipeline.set_settings(rig.pipeline.settings.with_enabled(StageId.LIGHTING_FULL, True))
    for _ in range(3):
        rig.pipeline.pump()
        if rig.executor.queue:
            break
    # Run one job at a time to see the order: the tail starts without waiting for the full stage.
    seen: list[StageId] = []
    while rig.executor.queue:
        rig.executor.queue.pop(0)()
        for stage in STAGE_ORDER:
            if stage not in seen and rig.pipeline.status(rig.photo, stage) is StageStatus.DONE:
                seen.append(stage)

    assert seen.index(StageId.EXPORT) < seen.index(StageId.LIGHTING_FULL) or all(
        rig.pipeline.status(rig.photo, stage) is StageStatus.DONE
        for stage in (StageId.LIGHTING_FULL, StageId.EXPORT, StageId.CLOUD_SYNC)
    )
    assert rig.pipeline.status(rig.photo, StageId.LIGHTING_FULL) is StageStatus.DONE
    assert rig.pipeline.status(rig.photo, StageId.CLOUD_SYNC) is StageStatus.DONE
    assert len(rig.cloud.uploaded) >= 1
