"""Tests for the neural cleanup stage's pure blending logic."""

import numpy as np

from photo_workflow.live.neural import blend_detail


def test_blend_keeps_network_chroma_and_mixes_luma() -> None:
    """Chroma comes from the network output while brightness stays near the original's."""
    original = np.full((32, 32, 3), (120, 100, 90), dtype=np.uint8)
    cleaned = np.full((32, 32, 3), (110, 100, 110), dtype=np.uint8)

    result = blend_detail(original, cleaned)

    assert result[..., 2].mean() > result[..., 0].mean() - 5  # cooler like the network
    assert abs(result.mean() - (original.mean() + cleaned.mean()) / 2) < 6
