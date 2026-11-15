"""Tests for the automatic crop scoring (fixed spec, worked examples)."""

from __future__ import annotations

import math

import pytest

from photo_workflow.live import autocrop_scoring as scoring

RATIOS = [3 / 2, 4 / 3, 5 / 4, 1, 4 / 5, 3 / 4, 2 / 3]


def window_from(choice: scoring.CropChoice) -> list[int]:
    """Return the window rounded like the spec table."""
    return [round(value) for value in choice.window]


@pytest.mark.parametrize(
    ("size", "subject", "ball", "direction", "ratio", "scale", "window", "score", "required"),
    [
        (  # A runner, no ball
            (6000, 4000),
            (2400, 1300, 3000, 2700),
            None,
            1,
            2 / 3,
            1.75,
            [2046, 965, 1895, 2842],
            0.941,
            [2340, 1188, 3060, 2812],
        ),
        (  # B soccer, ball ahead
            (6000, 4000),
            (2000, 1200, 2600, 2800),
            (3000, 2600, 3180, 2780),
            1,
            3 / 4,
            1.75,
            [1387, 627, 2436, 3248],
            0.957,
            [1940, 1072, 3270, 2928],
        ),
        (  # C unknown direction
            (6000, 4000),
            (2700, 1500, 3300, 2900),
            None,
            0,
            3 / 4,
            1.75,
            None,
            0.998,
            [2640, 1388, 3360, 3012],
        ),
        (  # D portrait jumper
            (4000, 6000),
            (1600, 1800, 2400, 3600),
            None,
            -1,
            3 / 4,
            1.3,
            [713, 1343, 2036, 2714],
            0.947,
            [1520, 1656, 2480, 3744],
        ),
        (  # E subject at the image edge
            (6000, 4000),
            (5200, 1400, 5900, 2800),
            None,
            -1,
            2 / 3,
            1.5,
            [4376, 1017, 1624, 2436],
            0.729,
            [5130, 1288, 5970, 2912],
        ),
    ],
)
def test_worked_examples_match_the_spec(
    size, subject, ball, direction, ratio, scale, window, score, required
) -> None:
    """The five worked examples of the spec reproduce (grid position within one step)."""
    choice = scoring.choose_crop(*size, subject, ball, direction, RATIOS)

    assert choice.ratio == pytest.approx(ratio)
    assert choice.scale == scale
    assert choice.score == pytest.approx(score, abs=0.002)
    assert [round(v) for v in choice.required] == required
    if window is not None:
        assert window_from(choice)[2:] == pytest.approx(window[2:], abs=1)
        assert window_from(choice)[:2] == pytest.approx(window[:2], abs=1)
    assert not choice.fallback


def test_example_a_rule_scores() -> None:
    """Rule scores of example A match the spec table."""
    choice = scoring.choose_crop(6000, 4000, (2400, 1300, 3000, 2700), None, 1, RATIOS)

    assert choice.rules == {
        "lead": 0.841,
        "size": 1.0,
        "thirds": 0.93,
        "head": 1.0,
        "edge": 1.0,
        "res": 1.0,
    }
    assert "ball" not in choice.rules


def test_single_rule_hand_checks() -> None:
    """Spec section 7 quick hand checks."""
    assert scoring.ramp(0.60, 0.15, 0.45, 0.75, 0.95) == 1
    assert scoring.ramp(0.30, 0.15, 0.45, 0.75, 0.95) == pytest.approx(0.5)
    assert scoring.ramp(0.85, 0.15, 0.45, 0.75, 0.95) == pytest.approx(0.5)
    assert scoring.ramp(0.13, 0, 0.08, 0.18, 0.40) == 1
    assert scoring.ramp(0.04, 0, 0.08, 0.18, 0.40) == pytest.approx(0.5)
    assert scoring.ramp(0.29, 0, 0.08, 0.18, 0.40) == pytest.approx(0.5)

    # Subject centre at 0.38 / 0.5 / 0.6 of a 1000 px wide window (d=+1).
    def lead(center: float) -> float:
        half = 50
        x = center * 1000
        _, rules = scoring.score_window(
            (0, 0, 1000, 1000), (x - half, 400, x + half, 600), None, 1, 1000, 1000
        )
        return rules["lead"]

    assert lead(0.38) == 1.0
    assert lead(0.5) == pytest.approx(0.4545, abs=1e-3)
    assert lead(0.6) == 0.0

    _, rules = scoring.score_window((100, 0, 1100, 1000), (120, 400, 220, 600), None, 0, 2000, 2000)
    assert rules["edge"] == pytest.approx(0.5)


def test_resolution_factor_penalises_small_windows() -> None:
    """A window with 10 % of the image area has a resolution factor of sqrt(0.5)."""
    side = math.sqrt(0.10 * 6000 * 4000)
    window = (0.0, 0.0, side, side)
    _, rules = scoring.score_window(window, (100, 100, 300, 300), None, 0, 6000, 4000)

    assert rules["res"] == pytest.approx(0.707, abs=1e-3)


def test_ball_dropped_when_far_from_subject() -> None:
    """A distant ball neither widens the crop nor takes part in scoring."""
    subject = (2000, 1500, 2600, 2500)
    near = scoring.choose_crop(6000, 4000, subject, (2700, 2300, 2800, 2400), 1, RATIOS)
    far = scoring.choose_crop(6000, 4000, subject, (5800, 100, 5900, 200), 1, RATIOS)
    without = scoring.choose_crop(6000, 4000, subject, None, 1, RATIOS)

    assert "ball" in near.rules
    assert "ball" not in far.rules
    assert far.window == without.window


def test_unknown_direction_is_taken_from_a_clearly_offset_ball() -> None:
    """With direction unknown, a ball well to one side sets it (even if too far to count)."""
    subject = (2000, 1500, 2600, 2500)

    assert scoring.resolve_direction(subject, (3200, 2000, 3300, 2100), 0) == 1
    assert scoring.resolve_direction(subject, (500, 2000, 600, 2100), 0) == -1
    assert scoring.resolve_direction(subject, (2300, 2000, 2400, 2100), 0) == 0
    assert scoring.resolve_direction(subject, (3200, 2000, 3300, 2100), -1) == -1


def test_result_ratio_is_always_one_of_the_allowed_ratios() -> None:
    """Only the allowed ratios are used, and the window always keeps the required rectangle."""
    allowed = [1.0, 4 / 5]
    choice = scoring.choose_crop(6000, 4000, (2400, 1300, 3000, 2700), None, 1, allowed)

    x, y, width, height = choice.window
    assert choice.ratio in allowed
    assert width / height == pytest.approx(choice.ratio)
    required = choice.required
    assert x <= required[0] + 1e-6
    assert y <= required[1] + 1e-6
    assert x + width >= required[2] - 1e-6
    assert y + height >= required[3] - 1e-6


def test_fallback_when_required_rectangle_is_larger_than_any_window() -> None:
    """A very wide subject with only a tall ratio falls back to the largest centred window."""
    choice = scoring.choose_crop(6000, 4000, (200, 1500, 5800, 2500), None, 0, [2 / 3])

    assert choice.fallback
    assert choice.ratio == pytest.approx(2 / 3)
    assert choice.window[3] == pytest.approx(4000)
    assert 0 <= choice.window[0] <= 6000 - choice.window[2]


def test_no_ratios_is_rejected() -> None:
    """At least one ratio is needed."""
    with pytest.raises(ValueError, match="aspect ratio"):
        scoring.choose_crop(6000, 4000, (1, 1, 2, 2), None, 0, [])
