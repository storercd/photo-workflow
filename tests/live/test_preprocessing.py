"""Tests for the Preprocessing stages: Denoise, Mask and Lighting/Color."""

from __future__ import annotations

from dataclasses import astuple, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from photo_workflow.editor.lighting import REFERENCE_KELVIN, estimate_adjustment
from photo_workflow.live import darktable, detection, imaging
from photo_workflow.live.api import StageJob, StageRegistry
from photo_workflow.live.detection import clear_detections, get_detection
from photo_workflow.live.features import denoise as denoise_stage
from photo_workflow.live.features import lighting as lighting_stage
from photo_workflow.live.features import load_features
from photo_workflow.live.features.denoise import MIN_STRENGTH, _strength, denoise_linear
from photo_workflow.live.features.mask import NO_SUBJECT_MESSAGE, apply_inverted_mask
from photo_workflow.live.imaging import (
    linear_to_srgb8,
    load_linear_rgb,
    load_srgb8,
    match_camera_look,
    save_linear_rgb,
)
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import StageId, StageStatus

PHOTO_NAME = "20251031_101500_00_IMG_0001.CR3"


@pytest.fixture(autouse=True)
def empty_cache() -> None:
    """Start every test with no cached detections."""
    clear_detections()


def make_job(
    tmp_path: Path,
    stage: StageId,
    source: Path,
    warnings: list[str],
    settings: LiveSettings | None = None,
) -> StageJob:
    """
    Build a job for a photo in a temporary Live folder.

    Returns:
        A job whose source is `source`.
    """
    layout = LiveLayout(tmp_path)
    photo = tmp_path / "2025" / "10" / "20251031" / "input" / PHOTO_NAME
    return StageJob(stage, photo, source, None, layout, settings or LiveSettings(), warnings.append)


def linear_gradient(height: int = 40, width: int = 60) -> np.ndarray:
    """
    Return a smooth linear-light test image.

    Returns:
        A 16-bit RGB array.
    """
    ramp = np.linspace(8000, 40000, width, dtype=np.float32)
    plane = np.tile(ramp, (height, 1))
    return np.stack((plane, plane * 0.9, plane * 0.8), axis=-1).astype(np.uint16)


def test_stages_are_registered_by_feature_discovery() -> None:
    """Dropping the modules into features/ is all the framework needs."""
    registry = StageRegistry()

    load_features(registry)

    assert all(
        registry.is_registered(stage) for stage in (StageId.DENOISE, StageId.MASK, StageId.LIGHTING)
    )


def test_linear_tiff_round_trip(tmp_path: Path) -> None:
    """Intermediates keep full 16-bit precision."""
    image = linear_gradient()
    path = tmp_path / "x" / "image.tif"

    save_linear_rgb(path, image)

    assert np.array_equal(load_linear_rgb(path), image)
    assert not list(path.parent.glob("*.partial*"))


def test_denoise_reduces_noise_and_keeps_the_signal() -> None:
    """Noise standard deviation drops while the mean level is preserved."""
    clean = linear_gradient(80, 80)
    noise = np.random.default_rng(1).normal(0, 1500, clean.shape)
    noisy = np.clip(clean + noise, 0, 65535).astype(np.uint16)

    result = denoise_linear(noisy)

    assert result.shape == noisy.shape and result.dtype == np.uint16
    error_before = np.std(noisy.astype(np.float32) - clean)
    error_after = np.std(result.astype(np.float32) - clean)
    assert error_after < error_before * 0.7
    assert abs(float(result.mean()) - float(clean.mean())) < 500


def test_denoise_removes_blotchy_colour_noise() -> None:
    """Colour speckle on a neutral grey is smoothed away."""
    rng = np.random.default_rng(3)
    grey = np.full((160, 160, 3), 20000, dtype=np.float32)
    speckle = cv2.GaussianBlur(rng.normal(0, 12000, grey.shape).astype(np.float32), (0, 0), 0.7)
    noisy = np.clip(grey + speckle, 0, 65535).astype(np.uint16)

    result = denoise_linear(noisy).astype(np.float32)

    def colour_spread(image: np.ndarray) -> float:
        return float(np.std(image[..., 0] - image[..., 2]))

    assert colour_spread(result) < colour_spread(noisy.astype(np.float32)) * 0.4


def test_denoise_strength_follows_the_noise_level() -> None:
    """A clean frame is barely smoothed; a noisy one gets a much stronger filter."""
    rng = np.random.default_rng(2)
    base = np.full((120, 120), 0.4, dtype=np.float32)
    quiet = base + rng.normal(0, 0.002, base.shape).astype(np.float32)
    noisy = base + rng.normal(0, 0.03, base.shape).astype(np.float32)

    assert _strength(quiet, 0.9) == MIN_STRENGTH
    assert 0.9 * 0.03 * 65535 * 0.8 < _strength(noisy, 0.9) < 0.9 * 0.03 * 65535 * 1.2


def test_darktable_sidecar_enables_profiled_denoise_with_valid_params() -> None:
    """The sidecar carries a full-size `denoiseprofile` parameter block in automatic mode."""
    assert len(darktable.denoise_params()) == 416
    sidecar = darktable.denoise_sidecar()
    assert 'darktable:operation="denoiseprofile"' in sidecar
    assert darktable.denoise_params().hex() in sidecar


def test_srgb_decoding_inverts_the_encoding() -> None:
    """Decoding darktable's sRGB export recovers the linear values."""
    linear = linear_gradient()

    decoded = darktable.srgb_to_linear(
        np.round(imaging._encode_srgb(linear) * 65535).astype(np.uint16)
    )

    assert np.abs(decoded.astype(int) - linear.astype(int)).max() <= 4


def test_denoise_stage_requires_darktable_for_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RAW goes through darktable; a missing darktable fails the stage."""
    raw = tmp_path / PHOTO_NAME
    raw.write_bytes(b"raw")
    linear = linear_gradient()
    monkeypatch.setattr(darktable, "find_cli", lambda: Path("/x/darktable-cli"))
    monkeypatch.setattr(denoise_stage, "denoise_raw", lambda _raw: linear)
    registry = StageRegistry()
    load_features(registry)
    warnings: list[str] = []

    result = registry.get(StageId.DENOISE).process(
        make_job(tmp_path, StageId.DENOISE, raw, warnings)
    )

    assert result.message == "darktable profiled denoise"
    assert np.array_equal(load_linear_rgb(result.outputs[-1]), linear)
    assert not warnings

    monkeypatch.setattr(imaging, "_develop_raw", lambda _path: linear)
    monkeypatch.setattr(
        denoise_stage,
        "denoise_raw",
        lambda _raw: (_ for _ in ()).throw(
            darktable.DarktableError("darktable-cli is not installed")
        ),
    )

    with pytest.raises(darktable.DarktableError):
        registry.get(StageId.DENOISE).process(make_job(tmp_path, StageId.DENOISE, raw, warnings))


def test_camera_look_follows_the_embedded_jpeg_tone_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Linear data is rendered like the camera JPEG, here a brighter, punchier curve."""
    linear = linear_gradient(80, 120)
    plain = linear_to_srgb8(linear)
    camera = np.clip(255 * (plain / 255.0) ** 0.7, 0, 255).astype(np.uint8)
    monkeypatch.setattr(imaging, "_embedded_preview", lambda _path: camera)

    matched = match_camera_look(linear, tmp_path / "x.CR3")

    assert matched is not None
    assert matched.shape == plain.shape
    assert np.abs(matched.astype(int) - camera.astype(int)).mean() < 2
    assert matched.mean() > plain.mean() + 10


def test_camera_look_is_skipped_without_a_matching_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or differently shaped preview falls back to the plain sRGB encoding."""
    linear = linear_gradient(80, 120)
    path = tmp_path / "image.tif"
    save_linear_rgb(path, linear)

    monkeypatch.setattr(imaging, "_embedded_preview", lambda _path: None)
    assert match_camera_look(linear, path) is None
    assert np.array_equal(load_srgb8(path, camera_reference=path), linear_to_srgb8(linear))

    portrait = np.zeros((120, 80, 3), np.uint8)
    monkeypatch.setattr(imaging, "_embedded_preview", lambda _path: portrait)
    assert match_camera_look(linear, path) is None


def test_inverted_mask_darkens_only_the_background_by_a_quarter_stop() -> None:
    """The subject is untouched; the background is scaled by 2**-0.25 in linear light."""
    image = np.full((100, 100, 3), 20000, dtype=np.uint16)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0

    result = apply_inverted_mask(image, mask)

    assert result[50, 50, 0] == 20000
    assert result[2, 2, 0] == pytest.approx(20000 * 2**-0.25, abs=2)


def test_mask_stage_writes_intermediate_and_publishes_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A found subject yields the masked intermediate and a cached subject box."""
    source = tmp_path / "denoise.tif"
    save_linear_rgb(source, np.full((40, 60, 3), 20000, dtype=np.uint16))

    def detector(rgb: np.ndarray) -> np.ndarray:
        mask = np.zeros(rgb.shape[:2], dtype=np.float32)
        mask[10:30, 20:40] = 1.0
        return mask

    monkeypatch.setattr(detection, "detect_subject_mask", detector)
    registry = StageRegistry()
    load_features(registry)
    warnings: list[str] = []
    job = make_job(tmp_path, StageId.MASK, source, warnings)

    result = registry.get(StageId.MASK).process(job)

    assert result.status is StageStatus.DONE
    output = result.outputs[-1]
    assert output.parent.name == "mask" and output.suffix == ".tif"
    assert load_linear_rgb(output)[0, 0, 0] < 20000
    found = get_detection(job.photo)
    assert found is not None and found.subject is not None
    assert astuple(found.subject) == pytest.approx((1 / 3, 0.25, 2 / 3, 0.75), abs=0.02)
    assert not warnings


def test_mask_stage_skips_and_flags_when_no_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a subject the stage is skipped without a warning and no stale output stays."""
    source = tmp_path / "denoise.tif"
    save_linear_rgb(source, np.full((40, 60, 3), 20000, dtype=np.uint16))
    monkeypatch.setattr(detection, "detect_subject_mask", lambda _rgb: None)
    registry = StageRegistry()
    load_features(registry)
    warnings: list[str] = []
    job = make_job(tmp_path, StageId.MASK, source, warnings)
    stale = job.output_path(suffix=".tif")
    stale.write_bytes(b"old")

    result = registry.get(StageId.MASK).process(job)

    assert result.status is StageStatus.SKIPPED and not result.outputs
    assert result.message == NO_SUBJECT_MESSAGE
    assert warnings == []
    assert not stale.exists()
    assert get_detection(job.photo) == detection.Detection()


def test_choose_adjustment_follows_the_settings() -> None:
    """Auto lighting and the temperature override can each be on or off."""
    dim = np.full((32, 32, 3), 40, dtype=np.uint8)
    auto = LiveSettings()

    assert lighting_stage.choose_adjustment(dim, auto, REFERENCE_KELVIN) == estimate_adjustment(
        dim, None, REFERENCE_KELVIN
    )
    off = replace(auto, auto_lighting=False)
    assert lighting_stage.choose_adjustment(dim, off, REFERENCE_KELVIN).is_neutral

    warm = replace(off, color_temperature=4000.0)
    only_white_balance = lighting_stage.choose_adjustment(dim, warm, REFERENCE_KELVIN)
    assert only_white_balance.exposure_ev == 0
    assert only_white_balance.temperature == pytest.approx(4000, rel=0.02)

    both = lighting_stage.choose_adjustment(dim, replace(auto, color_temperature=4000.0), 6500.0)
    assert both == estimate_adjustment(dim, 4000.0, 6500.0)
    assert both.exposure_ev > 0


def test_lighting_stage_writes_the_final_jpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage develops its source, adjusts it and writes a high-quality JPEG to output/."""
    source = tmp_path / "mask.tif"
    dark = np.full((40, 60, 3), 3000, dtype=np.uint16)
    save_linear_rgb(source, dark)
    monkeypatch.setattr(lighting_stage, "as_shot_reference", lambda _photo: REFERENCE_KELVIN)
    registry = StageRegistry()
    load_features(registry)
    job = make_job(tmp_path, StageId.LIGHTING, source, [])

    result = registry.get(StageId.LIGHTING).process(job)

    assert result.status is StageStatus.DONE
    output = result.outputs[-1]
    assert output.parent.name == "output" and output.suffix == ".jpg"
    written = cv2.imread(str(output))
    assert written.shape == (40, 60, 3)
    assert written.mean() > linear_to_srgb8(dark).mean()


def test_as_shot_reference_falls_back_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing metadata uses the editor's reference temperature; real values are clamped."""
    monkeypatch.setattr(lighting_stage, "read_as_shot_kelvin", lambda _p: None)
    assert lighting_stage.as_shot_reference(Path("x")) == REFERENCE_KELVIN

    def fail(_path: Path) -> float:
        raise RuntimeError("no exiftool")

    monkeypatch.setattr(lighting_stage, "read_as_shot_kelvin", fail)
    assert lighting_stage.as_shot_reference(Path("x")) == REFERENCE_KELVIN

    monkeypatch.setattr(lighting_stage, "read_as_shot_kelvin", lambda _p: 99999.0)
    assert lighting_stage.as_shot_reference(Path("x")) == lighting_stage.MAX_KELVIN
