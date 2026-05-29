from __future__ import annotations

from datetime import date
from pathlib import Path

from photo_workflow import config as app_config
from photo_workflow import video_notes


def test_build_today_source_dir_uses_expected_dated_path() -> None:
    source_dir = app_config.build_today_source_dir(
        today=date(2026, 5, 28),
        camera_root=Path("/tmp/camera"),
    )

    assert source_dir == Path("/tmp/camera/20260528")


def test_load_config_returns_defaults_when_config_is_missing(tmp_path: Path) -> None:
    config = app_config.load_config(tmp_path / "missing.toml")

    assert config.workflow.camera_root == app_config.DEFAULT_CAMERA_ROOT
    assert (
        config.memory_card_copy.copy_verification == app_config.DEFAULT_COPY_VERIFICATION
    )
    assert (
        config.memory_card_copy.ignored_extensions
        == app_config.DEFAULT_IGNORED_CARD_EXTENSIONS
    )
    assert config.video_notes.max_duration_seconds == app_config.DEFAULT_MAX_DURATION_SECONDS


def test_load_config_reads_sections_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text(
        '[workflow]\ncamera_root = "~/camera-roll"\n\n'
        '[memory_card_copy]\ncopy_verification = "crc32"\n'
        'ignored_extensions = ["ctg", ".LOG"]\n\n'
        '[video_notes]\nmax_duration_seconds = 7.5\n'
    )

    config = app_config.load_config(config_path)

    assert config.workflow.camera_root == Path("~/camera-roll").expanduser()
    assert config.memory_card_copy.copy_verification == "crc32"
    assert config.memory_card_copy.ignored_extensions == (".ctg", ".log")
    assert config.video_notes.max_duration_seconds == 7.5


def test_build_today_source_dir_uses_configured_camera_root(tmp_path: Path, monkeypatch) -> None:
    configured_root = tmp_path / "camera"
    monkeypatch.setattr(
        app_config,
        "load_workflow_config",
        lambda config_path=app_config.DEFAULT_CONFIG_PATH: app_config.WorkflowConfig(
            camera_root=configured_root,
        ),
    )

    source_dir = app_config.build_today_source_dir(today=date(2026, 5, 29))

    assert source_dir == configured_root / "20260529"


def test_process_short_videos_creates_note_for_short_mp4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "clip.MP4"
    video_path.write_bytes(b"video")
    expected_timestamp = 1_717_171_717_000_000_000
    video_notes.os.utime(video_path, ns=(expected_timestamp, expected_timestamp))

    monkeypatch.setattr(video_notes, "probe_video_duration_seconds", lambda _: 4.2)
    monkeypatch.setattr(video_notes, "transcribe_video", lambda *args, **kwargs: "hello world")

    def fake_create_note_image(output_path: Path, transcription: str) -> None:
        output_path.write_text(transcription)

    monkeypatch.setattr(video_notes, "create_note_image", fake_create_note_image)

    processed = video_notes.process_short_videos(tmp_path)

    output_path = tmp_path / "clip.tif"
    assert [item.output_path for item in processed] == [output_path]
    assert output_path.read_text() == "hello world"
    assert output_path.stat().st_mtime_ns == expected_timestamp


def test_process_short_videos_skips_long_mp4_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "long.MP4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(video_notes, "probe_video_duration_seconds", lambda _: 10.0)
    monkeypatch.setattr(video_notes, "transcribe_video", lambda *args, **kwargs: "unused")

    processed = video_notes.process_short_videos(tmp_path, max_duration_seconds=10.0)

    assert processed == []
    assert not (tmp_path / "long.tif").exists()


def test_process_short_videos_uses_configured_max_duration_when_not_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "clip.MP4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(video_notes, "probe_video_duration_seconds", lambda _: 8.0)
    monkeypatch.setattr(
        video_notes,
        "load_video_notes_config",
        lambda config_path=app_config.DEFAULT_CONFIG_PATH: app_config.VideoNotesConfig(
            max_duration_seconds=9.0,
        ),
    )
    monkeypatch.setattr(video_notes, "transcribe_video", lambda *args, **kwargs: "hello world")

    def fake_create_note_image(output_path: Path, transcription: str) -> None:
        output_path.write_text(transcription)

    monkeypatch.setattr(video_notes, "create_note_image", fake_create_note_image)

    processed = video_notes.process_short_videos(tmp_path)

    assert [item.output_path.name for item in processed] == ["clip.tif"]


def test_main_reports_when_no_short_videos_are_processed(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        video_notes,
        "build_today_source_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        video_notes,
        "load_video_notes_config",
        lambda config_path=app_config.DEFAULT_CONFIG_PATH: app_config.VideoNotesConfig(
            max_duration_seconds=12.5,
        ),
    )
    monkeypatch.setattr(
        video_notes,
        "process_short_videos",
        lambda source_dir=None, max_duration_seconds=None: [],
    )
    (tmp_path / "long.MP4").write_bytes(b"video")

    with caplog.at_level("INFO"):
        video_notes.main()

    assert "found 1 .mp4 file(s)" in caplog.text
    assert "none were shorter than 12.5 seconds" in caplog.text


def test_iter_mp4_files_matches_case_insensitive_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.MP4").write_bytes(b"video")
    (tmp_path / "b.mp4").write_bytes(b"video")
    (tmp_path / "c.Mov").write_bytes(b"video")

    matched = video_notes.iter_mp4_files(tmp_path)

    assert [path.name for path in matched] == ["a.MP4", "b.mp4"]


def test_build_center_text_layout_uses_larger_font_for_shorter_text() -> None:
    short_font, _, _ = video_notes.build_center_text_layout(
        "Peach",
        image_size=(video_notes.DEFAULT_NOTE_WIDTH, video_notes.DEFAULT_NOTE_HEIGHT),
        border_width=video_notes.DEFAULT_NOTE_BORDER_WIDTH,
    )
    long_font, _, _ = video_notes.build_center_text_layout(
        "Trying to announce it more clearly this time while also describing focus, "
        "exposure, shutter speed, framing, and the intended composition for later review.",
        image_size=(video_notes.DEFAULT_NOTE_WIDTH, video_notes.DEFAULT_NOTE_HEIGHT),
        border_width=video_notes.DEFAULT_NOTE_BORDER_WIDTH,
    )

    assert short_font.size > long_font.size


def test_build_center_text_layout_applies_text_scale_factor(monkeypatch) -> None:
    sample_text = "Out of focus, I'm a point."

    monkeypatch.setattr(video_notes, "DEFAULT_TEXT_SCALE_FACTOR", 0.5)
    half_font, _, _ = video_notes.build_center_text_layout(
        sample_text,
        image_size=(video_notes.DEFAULT_NOTE_WIDTH, video_notes.DEFAULT_NOTE_HEIGHT),
        border_width=video_notes.DEFAULT_NOTE_BORDER_WIDTH,
    )

    monkeypatch.setattr(video_notes, "DEFAULT_TEXT_SCALE_FACTOR", 0.75)
    larger_font, _, _ = video_notes.build_center_text_layout(
        sample_text,
        image_size=(video_notes.DEFAULT_NOTE_WIDTH, video_notes.DEFAULT_NOTE_HEIGHT),
        border_width=video_notes.DEFAULT_NOTE_BORDER_WIDTH,
    )

    assert half_font.size < larger_font.size


def test_iter_note_font_paths_prefers_configured_then_system_then_cached_fonts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    configured_font_path = tmp_path / "configured.ttf"
    cached_font_path = tmp_path / "cache" / video_notes.DEFAULT_DOWNLOADED_FONT_FILENAME
    configured_font_path.write_bytes(b"configured")
    cached_font_path.parent.mkdir(parents=True)
    cached_font_path.write_bytes(b"cached")

    monkeypatch.setenv(video_notes.DEFAULT_FONT_PATH_ENV_VAR, str(configured_font_path))
    monkeypatch.setenv(video_notes.DEFAULT_FONT_CACHE_ENV_VAR, str(cached_font_path.parent))

    font_paths = video_notes.iter_note_font_paths()

    assert font_paths[0] == configured_font_path
    assert Path(video_notes.DEFAULT_FONT_PATHS[0]) in font_paths[1:-1]
    assert font_paths[-1] == cached_font_path


def test_install_default_note_font_downloads_into_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / video_notes.DEFAULT_DOWNLOADED_FONT_FILENAME

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self.payload

    monkeypatch.setattr(
        video_notes.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(b"font-data"),
    )

    installed_path = video_notes.install_default_note_font(destination=destination)

    assert installed_path == destination
    assert destination.read_bytes() == b"font-data"
