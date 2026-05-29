from photo_workflow import __version__


def test_exposes_version() -> None:
    assert __version__ == "0.1.0"
