"""Package metadata tests."""

from photo_workflow import __version__


def test_exposes_version() -> None:
    """Verify the package exposes the expected version string."""
    assert __version__ == "0.1.0"
