"""Tests for the shared subject and ball detection module."""

from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

import cv2
import numpy as np
import pytest

from photo_workflow.editor.ball import BallCandidate
from photo_workflow.live import detection
from photo_workflow.live.detection import (
    BoundingBox,
    Detection,
    analyze_image,
    clear_detections,
    detect_photo,
    get_detection,
    store_detection,
)


@pytest.fixture(autouse=True)
def empty_cache() -> None:
    """Start every test with no cached detections."""
    clear_detections()


def square_detector(rgb: np.ndarray) -> np.ndarray:
    """
    Pretend the subject is a rectangle covering the middle of the image.

    Returns:
        A float mask the size of the image.
    """
    mask = np.zeros(rgb.shape[:2], dtype=np.float32)
    height, width = mask.shape
    mask[height // 4 : height // 2, width // 5 : width // 2] = 1.0
    return mask


def test_bounding_box_geometry() -> None:
    """Boxes report size and center and convert to pixels."""
    box = BoundingBox(0.2, 0.25, 0.5, 0.75)

    assert box.width == pytest.approx(0.3)
    assert box.height == pytest.approx(0.5)
    assert box.center == pytest.approx((0.35, 0.5))
    assert box.to_pixels(1000, 400) == (200, 100, 500, 300)


def test_analyze_image_reports_normalized_subject_box_and_mask() -> None:
    """The subject box is relative to the image and the mask comes back for the caller."""
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    analysis = analyze_image(image, detector=square_detector)

    assert analysis.detection.subject is not None
    assert astuple(analysis.detection.subject) == pytest.approx((0.2, 0.25, 0.5, 0.5), abs=0.01)
    assert analysis.detection.ball is None
    assert analysis.mask is not None
    assert analysis.mask.shape == (400, 600)


def test_analyze_image_without_subject() -> None:
    """No subject means no box and no mask."""
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    analysis = analyze_image(image, detector=lambda _rgb: None)

    assert analysis.detection == Detection()
    assert not analysis.detection.subject_found
    assert analysis.mask is None


def test_analyze_image_downscales_large_images() -> None:
    """Detection runs on a reduced copy but boxes stay normalized."""
    image = np.zeros((3200, 4800, 3), dtype=np.uint8)
    seen: list[tuple[int, ...]] = []

    def detector(rgb: np.ndarray) -> np.ndarray:
        seen.append(rgb.shape)
        return square_detector(rgb)

    analysis = analyze_image(image, detector=detector)

    assert max(seen[0][:2]) == detection.WORKING_LONG_EDGE
    assert analysis.detection.subject is not None
    assert astuple(analysis.detection.subject) == pytest.approx((0.2, 0.25, 0.5, 0.5), abs=0.01)


def test_combine_instances_keeps_comparable_instances_only() -> None:
    """Small instances are noise; players of similar size form one subject."""
    big, similar, tiny = (np.zeros((20, 20), dtype=np.float32) for _ in range(3))
    big[:10, :10] = 1
    similar[10:, 10:16] = 1
    tiny[19, 19] = 1

    combined = detection._combine_instances([big, similar, tiny])

    assert combined is not None
    assert combined[0, 0] == 1 and combined[15, 12] == 1 and combined[19, 19] == 0
    assert detection._combine_instances([tiny]) is None
    assert detection._combine_instances([]) is None


def test_cache_is_keyed_by_photo_and_clearable() -> None:
    """Stored detections are returned for the same photo path only."""
    photo, other = Path("a/input/one.CR3"), Path("a/input/two.CR3")
    found = Detection(BoundingBox(0, 0, 0.5, 0.5))

    store_detection(photo, found)

    assert get_detection(photo) is found
    assert get_detection(other) is None
    clear_detections(photo)
    assert get_detection(photo) is None


def test_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The oldest entries are dropped once the cache is full."""
    monkeypatch.setattr(detection, "CACHE_SIZE", 2)

    for name in ("a", "b", "c"):
        store_detection(Path(name), Detection())

    assert get_detection(Path("a")) is None
    assert get_detection(Path("c")) is not None


def test_detect_photo_runs_once_then_uses_the_cache(tmp_path: Path) -> None:
    """Automatic Cropping can call detect_photo with any pipeline output."""
    source = tmp_path / "preview.jpg"
    cv2.imwrite(str(source), np.full((60, 90, 3), 128, dtype=np.uint8))
    calls: list[int] = []

    def detector(rgb: np.ndarray) -> np.ndarray:
        calls.append(1)
        return square_detector(rgb)

    photo = tmp_path / "input" / "x.CR3"
    first = detect_photo(photo, source, detector=detector)
    second = detect_photo(photo, source, detector=detector)

    assert first is second and first.subject_found
    assert len(calls) == 1


def test_ball_box_is_found_and_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ball box is built from the editor's candidate circle."""
    monkeypatch.setattr(
        detection, "detect_ball_candidates", lambda *_a, **_k: [BallCandidate(0.5, 0.5, 0.1)]
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    ball = analyze_image(image, detector=lambda _rgb: None).detection.ball

    assert ball is not None
    assert astuple(ball) == pytest.approx((0.4, 0.3, 0.6, 0.7))
