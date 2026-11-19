"""Shared fixtures for the Live-mode tests."""

import pytest

from photo_workflow.live import neural


@pytest.fixture(autouse=True)
def skip_neural_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep stage tests fast and offline: the real network is covered by a dedicated test."""
    monkeypatch.setattr(neural, "clean", lambda rgb: rgb)
