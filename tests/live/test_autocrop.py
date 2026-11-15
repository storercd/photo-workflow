"""Tests for the AutoCrop stage and the crop-source marker."""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

import pytest
from test_pipeline import ManualExecutor, make_photo

from photo_workflow.editor import metadata as crop_metadata
from photo_workflow.editor.metadata import PhotoMetadata
from photo_workflow.editor.model import CropRect
from photo_workflow.live.api import StageImplementation, StageRegistry
from photo_workflow.live.features import autocrop
from photo_workflow.live.features.autocrop import AutoCrop, Detection
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.pipeline import LivePipeline
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import StageId, StageStatus

WIDTH, HEIGHT = 6000, 4000
HAS_EXIFTOOL = shutil.which("exiftool") is not None


class FakeDetector:
    """Returns a fixed detection and records how it was called."""

    def __init__(self, detection: Detection | None) -> None:
        """Remember the detection to return."""
        self.detection = detection
        self.calls: list[tuple[Path, bool]] = []

    def detect(self, image: Path, *, want_ball: bool) -> Detection | None:
        """
        Record the call.

        Returns:
            The fixed detection.
        """
        self.calls.append((image, want_ball))
        return self.detection


RUNNER = Detection(
    WIDTH, HEIGHT, (2400, 1300, 3000, 2700), ball=(3000, 2600, 3180, 2780), direction=1
)


class StageHarness:
    """Runs AutoCrop through the real pipeline with metadata and writing stubbed."""

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detection: Detection | None
    ) -> None:
        """Wire the stage into a pipeline over a temporary Live folder."""
        self.photo = make_photo(tmp_path)
        self.detector = FakeDetector(detection)
        self.writes: list[tuple[CropRect, str]] = []
        self.manual = False
        monkeypatch.setattr(
            autocrop,
            "read_photo_metadata",
            lambda *_: PhotoMetadata(WIDTH, HEIGHT, CropRect(0, 0, 1, 1)),
        )
        monkeypatch.setattr(autocrop, "is_manual_crop", lambda _: self.manual)
        monkeypatch.setattr(
            autocrop,
            "write_photo_crop",
            lambda _raw, _xmp, crop, *, source: self.writes.append((crop, source)),
        )
        registry = StageRegistry()
        registry.register(StageId.AUTOCROP, AutoCrop(self.detector))
        self.executor = ManualExecutor()
        self.pipeline = LivePipeline(
            LiveLayout(tmp_path),
            LiveSettings(autocrop_ratios=("3:2", "4:3", "1:1", "4:5", "3:4", "2:3")),
            registry,
            executor=self.executor,
            sleep=lambda _: None,
            backoff=(),
        )
        self.pipeline.scan()

    def run(self) -> None:
        """Process everything queued."""
        self.executor.run_all()

    def settings(self, **changes: object) -> None:
        """Apply changed Live settings."""
        self.pipeline.set_settings(replace(self.pipeline.settings, **changes))


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StageHarness:
    """
    A harness whose detector finds the runner.

    Returns:
        The harness.
    """
    return StageHarness(tmp_path, monkeypatch, RUNNER)


def test_writes_an_automatic_crop_within_the_allowed_ratios(harness: StageHarness) -> None:
    """The crop is written as auto and its displayed ratio is an allowed one."""
    harness.run()

    ((crop, source),) = harness.writes
    assert source == "auto"
    ratio = (crop.width * WIDTH) / (crop.height * HEIGHT)
    assert any(
        ratio == pytest.approx(value, rel=1e-6) for value in (3 / 2, 4 / 3, 1, 4 / 5, 3 / 4, 2 / 3)
    )
    assert harness.pipeline.status(harness.photo, StageId.AUTOCROP) is StageStatus.DONE


def test_ball_is_only_requested_when_the_setting_is_on(harness: StageHarness) -> None:
    """Ball detection is off by default and on with the ball option."""
    harness.run()
    harness.settings(autocrop_ball=True)
    harness.run()

    assert [want_ball for _, want_ball in harness.detector.calls] == [False, True]


def test_manual_crop_is_kept_unless_forced(harness: StageHarness) -> None:
    """A manual crop is never overwritten, until Force automatic crop is on."""
    harness.manual = True
    harness.run()

    assert harness.writes == []
    assert len(harness.detector.calls) == 1  # still computed

    harness.settings(autocrop_force=True)
    harness.run()

    assert len(harness.writes) == 1
    assert harness.writes[0][1] == "auto"


def test_no_subject_skips_with_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a detection nothing is written and a warning is recorded."""
    harness = StageHarness(tmp_path, monkeypatch, None)

    harness.run()

    assert harness.writes == []
    assert harness.pipeline.status(harness.photo, StageId.AUTOCROP) is StageStatus.SKIPPED
    assert "no subject" in harness.pipeline.warnings()[0].message


def test_missing_detection_module_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no live.detection module the default detector finds nothing."""

    def fail(_: str) -> object:
        raise ImportError

    monkeypatch.setattr(autocrop.importlib, "import_module", fail)

    assert autocrop.default_detector().detect(Path("x.cr3"), want_ball=True) is None


def test_detection_is_scaled_from_a_preview_to_the_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boxes found on a smaller preview map to the same normalized crop."""
    full = autocrop.compute_crop(RUNNER, WIDTH, HEIGHT, [1.0, 2 / 3], use_ball=False)[0]
    half = Detection(3000, 2000, (1200, 650, 1500, 1350), direction=1)

    scaled = autocrop.compute_crop(half, WIDTH, HEIGHT, [1.0, 2 / 3], use_ball=False)[0]

    assert scaled.left == pytest.approx(full.left)
    assert scaled.bottom == pytest.approx(full.bottom)


def test_changing_autocrop_options_reruns_every_photo(harness: StageHarness) -> None:
    """Force, ball and ratio changes invalidate AutoCrop for all photos."""
    harness.run()
    assert len(harness.writes) == 1

    harness.settings(autocrop_ratios=("1:1",))
    harness.run()

    assert len(harness.writes) == 2


def test_stage_is_a_registered_implementation() -> None:
    """The module registers itself for the AutoCrop stage."""
    registry = StageRegistry()
    autocrop.register(registry)

    assert registry.is_registered(StageId.AUTOCROP)
    assert isinstance(registry.get(StageId.AUTOCROP), StageImplementation)


def test_new_sidecar_records_the_crop_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sidecar created by the writer carries pwf:CropSource."""
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, '[{"ImageWidth": 6000, "ImageHeight": 4000}]'
        ),
    )
    xmp_path = tmp_path / "photo.xmp"
    crop = CropRect(0.1, 0.1, 0.9, 0.9)

    crop_metadata.write_photo_crop(tmp_path / "photo.cr3", xmp_path, crop, source="auto")

    description = (
        ElementTree
        .parse(xmp_path)
        .getroot()
        .find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    )
    assert description is not None
    assert description.get(f"{{{crop_metadata.PWF_NS}}}CropSource") == "auto"


def test_existing_sidecar_write_defaults_to_a_manual_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Crop mode's write path marks crops as manual without any change on its side."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, '[{"ImageWidth":6000,"ImageHeight":4000,"Orientation":1}]'
        )

    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(crop_metadata.subprocess, "run", run)
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()

    crop_metadata.write_photo_crop(tmp_path / "photo.cr3", xmp_path, CropRect(0.1, 0.1, 0.9, 0.9))

    assert "-XMP-pwf:CropSource=manual" in calls[-1]
    assert "-config" in calls[-1]


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="ExifTool is not installed")
def test_is_manual_crop_reads_the_marker_with_exiftool(tmp_path: Path) -> None:
    """Manual unless marked auto; a sidecar without any crop is not manual."""
    header = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        f'xmlns:pwf="{crop_metadata.PWF_NS}" '
    )
    cases = {
        "none": ('xmp:Rating="3"', False),
        "legacy": ('crs:CropLeft="0.1" crs:HasCrop="True"', True),
        "auto": ('crs:CropLeft="0.1" crs:HasCrop="True" pwf:CropSource="auto"', False),
        "manual": ('crs:CropLeft="0.1" crs:HasCrop="True" pwf:CropSource="manual"', True),
    }
    for name, (attributes, expected) in cases.items():
        xmp_path = tmp_path / f"{name}.xmp"
        xmp_path.write_text(f"{header}{attributes}/></rdf:RDF></x:xmpmeta>")

        assert crop_metadata.is_manual_crop(xmp_path) is expected, name

    assert crop_metadata.is_manual_crop(tmp_path / "missing.xmp") is False


@pytest.mark.skipif(not HAS_EXIFTOOL, reason="ExifTool is not installed")
def test_exiftool_updates_an_existing_sidecar_marker(tmp_path: Path) -> None:
    """Writing a manual crop over an automatic one flips the marker (real ExifTool)."""
    xmp_path = tmp_path / "photo.xmp"
    subprocess.run(
        [
            "exiftool",
            "-config",
            str(crop_metadata.PWF_CONFIG),
            "-XMP-crs:HasCrop=True",
            "-XMP-pwf:CropSource=auto",
            str(xmp_path),
        ],
        check=True,
        capture_output=True,
    )
    assert crop_metadata.is_manual_crop(xmp_path) is False

    subprocess.run(
        [
            "exiftool",
            "-config",
            str(crop_metadata.PWF_CONFIG),
            "-XMP-pwf:CropSource=manual",
            str(xmp_path),
        ],
        check=True,
        capture_output=True,
    )

    assert crop_metadata.is_manual_crop(xmp_path) is True
