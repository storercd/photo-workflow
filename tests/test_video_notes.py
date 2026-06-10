"""Video note and shared config tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from photo_workflow import config as app_config
from photo_workflow import video_notes


def test_build_today_source_dir_uses_expected_dated_path() -> None:
    """Verify the dated source directory format uses YYYYMMDD."""
    source_dir = app_config.build_today_source_dir(
        today=date(2026, 5, 28),
        camera_root=Path("/tmp/camera"),
    )

    assert source_dir == Path("/tmp/camera/20260528")


def test_load_config_returns_defaults_when_config_is_missing(tmp_path: Path) -> None:
    """Verify missing config files fall back to default settings."""
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
    assert config.video_notes.transcription_model == app_config.DEFAULT_TRANSCRIPTION_MODEL


def test_load_config_reads_sections_from_toml(tmp_path: Path) -> None:
    """Verify TOML settings are loaded and normalized across config sections."""
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text(
        '[workflow]\ncamera_root = "~/camera-roll"\n\n'
        '[memory_card_copy]\ncopy_verification = "crc32"\n'
        'ignored_extensions = ["ctg", ".LOG"]\n\n'
        '[video_notes]\nmax_duration_seconds = 7.5\n'
        'transcription_model = "mlx-community/whisper-medium-mlx"\n'
    )

    config = app_config.load_config(config_path)

    assert config.workflow.camera_root == Path("~/camera-roll").expanduser()
    assert config.memory_card_copy.copy_verification == "crc32"
    assert config.memory_card_copy.ignored_extensions == (".ctg", ".log")
    assert config.video_notes.max_duration_seconds == 7.5
    assert config.video_notes.transcription_model == "mlx-community/whisper-medium-mlx"


def test_build_today_source_dir_uses_configured_camera_root(tmp_path: Path, monkeypatch) -> None:
    """Verify the shared workflow camera root controls the dated source path."""
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
    """Verify short clips generate note images with copied timestamps."""
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


def test_process_short_videos_can_write_notes_outside_video_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify note images can be written to a separate target folder."""
    video_dir = tmp_path / "videos"
    output_dir = tmp_path / "processed"
    video_dir.mkdir()
    output_dir.mkdir()
    video_path = video_dir / "clip.MP4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(video_notes, "probe_video_duration_seconds", lambda _: 4.2)
    monkeypatch.setattr(video_notes, "transcribe_video", lambda *args, **kwargs: "hello world")

    def fake_create_note_image(output_path: Path, transcription: str) -> None:
        output_path.write_text(transcription)

    monkeypatch.setattr(video_notes, "create_note_image", fake_create_note_image)

    processed = video_notes.process_short_videos(video_dir, output_dir=output_dir)

    assert [item.video_path for item in processed] == [video_path]
    assert [item.output_path for item in processed] == [output_dir / "clip.tif"]


def test_move_videos_to_processing_subdir_moves_only_top_level_mp4_files(tmp_path: Path) -> None:
    """Verify top-level MP4 files are relocated into the videos subfolder."""
    top_level_video = tmp_path / "clip.MP4"
    existing_video_dir = tmp_path / video_notes.DEFAULT_VIDEO_SUBDIR_NAME
    nested_video = existing_video_dir / "nested.mp4"
    image_path = tmp_path / "photo.jpg"
    top_level_video.write_bytes(b"video")
    existing_video_dir.mkdir()
    nested_video.write_bytes(b"video")
    image_path.write_bytes(b"image")

    video_dir = video_notes.move_videos_to_processing_subdir(tmp_path)

    assert video_dir == existing_video_dir
    assert not top_level_video.exists()
    assert (video_dir / "clip.MP4").exists()
    assert nested_video.exists()
    assert image_path.exists()


def test_move_videos_to_processing_subdir_skips_folder_creation_without_top_level_mp4s(
    tmp_path: Path,
) -> None:
    """Verify the videos subfolder is not created when no top-level MP4 files exist."""
    existing_video_dir = tmp_path / video_notes.DEFAULT_VIDEO_SUBDIR_NAME
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"image")

    video_dir = video_notes.move_videos_to_processing_subdir(tmp_path)

    assert video_dir == existing_video_dir
    assert not existing_video_dir.exists()
    assert image_path.exists()


def test_process_short_videos_skips_long_mp4_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify clips at or above the duration limit are skipped."""
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
    """Verify video processing uses the configured duration limit by default."""
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


def test_process_short_videos_uses_configured_transcription_model_when_not_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify video processing uses the configured transcription model by default."""
    video_path = tmp_path / "clip.MP4"
    video_path.write_bytes(b"video")
    captured_models: list[str] = []

    monkeypatch.setattr(video_notes, "probe_video_duration_seconds", lambda _: 4.0)
    monkeypatch.setattr(
        video_notes,
        "load_video_notes_config",
        lambda config_path=app_config.DEFAULT_CONFIG_PATH: app_config.VideoNotesConfig(
            max_duration_seconds=9.0,
            transcription_model="mlx-community/whisper-medium-mlx",
        ),
    )

    def fake_transcribe_video(video_path: Path, *, model: str) -> str:
        captured_models.append(model)
        return "hello world"

    def fake_create_note_image(output_path: Path, transcription: str) -> None:
        output_path.write_text(transcription)

    monkeypatch.setattr(video_notes, "transcribe_video", fake_transcribe_video)
    monkeypatch.setattr(video_notes, "create_note_image", fake_create_note_image)

    processed = video_notes.process_short_videos(tmp_path)

    assert [item.output_path.name for item in processed] == ["clip.tif"]
    assert captured_models == ["mlx-community/whisper-medium-mlx"]


def test_main_reports_when_no_short_videos_are_processed(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify the CLI logs when no videos qualify for note generation."""
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
            transcription_model="mlx-community/whisper-medium-mlx",
        ),
    )
    monkeypatch.setattr(
        video_notes,
        "process_short_videos",
        lambda source_dir=None, output_dir=None, max_duration_seconds=None, transcription_model=None: [],
    )
    (tmp_path / "long.MP4").write_bytes(b"video")

    with caplog.at_level("INFO"):
        video_notes.main()

    assert "found 1 .mp4 file(s)" in caplog.text
    assert "none were shorter than 12.5 seconds" in caplog.text


def test_run_video_notes_step_logs_transcription_model(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify the video notes step logs the configured transcription model."""
    monkeypatch.setattr(
        video_notes,
        "process_short_videos",
        lambda source_dir=None, output_dir=None, max_duration_seconds=None, transcription_model=None: [],
    )

    with caplog.at_level("INFO"):
        video_notes.run_video_notes_step(
            tmp_path,
            config=app_config.VideoNotesConfig(
                max_duration_seconds=10.0,
                transcription_model="mlx-community/whisper-medium-mlx",
            ),
        )

    assert "using transcription model mlx-community/whisper-medium-mlx" in caplog.text


def test_run_video_notes_step_logs_when_no_videos_are_present(
    tmp_path: Path,
    caplog,
) -> None:
    """Verify the workflow does not fail when there are no MP4 files to process."""
    with caplog.at_level("INFO"):
        video_notes.run_video_notes_step(
            tmp_path,
            config=app_config.VideoNotesConfig(max_duration_seconds=10.0),
        )

    assert "no .mp4 files found" in caplog.text


def test_run_video_notes_step_moves_videos_before_processing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the workflow relocates videos before creating note images."""
    video_path = tmp_path / "clip.MP4"
    video_path.write_bytes(b"video")
    captured_calls: list[tuple[Path, Path | None]] = []

    monkeypatch.setattr(
        video_notes,
        "process_short_videos",
        lambda source_dir=None, output_dir=None, max_duration_seconds=None, transcription_model=None: captured_calls.append(
            (source_dir, output_dir)
        ) or [],
    )

    video_notes.run_video_notes_step(
        tmp_path,
        config=app_config.VideoNotesConfig(max_duration_seconds=10.0),
    )

    assert not video_path.exists()
    assert (tmp_path / video_notes.DEFAULT_VIDEO_SUBDIR_NAME / "clip.MP4").exists()
    assert captured_calls == [
        (
            tmp_path / video_notes.DEFAULT_VIDEO_SUBDIR_NAME,
            tmp_path,
        )
    ]


def test_benchmark_transcriptions_returns_results_for_each_file_and_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify transcription benchmarking runs every requested model for each MP4 clip."""
    (tmp_path / "clip1.MP4").write_bytes(b"video")
    (tmp_path / "clip2.mp4").write_bytes(b"video")
    recorded_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        video_notes,
        "transcribe_video",
        lambda video_path, *, model: recorded_calls.append((video_path.name, model))
        or f"{video_path.name}:{model}",
    )

    results = video_notes.benchmark_transcriptions(
        tmp_path,
        models=("mlx-community/whisper-tiny-mlx", "mlx-community/whisper-medium-mlx"),
    )

    assert [(result.video_path.name, result.model) for result in results] == [
        ("clip1.MP4", "mlx-community/whisper-tiny-mlx"),
        ("clip1.MP4", "mlx-community/whisper-medium-mlx"),
        ("clip2.mp4", "mlx-community/whisper-tiny-mlx"),
        ("clip2.mp4", "mlx-community/whisper-medium-mlx"),
    ]
    assert recorded_calls == [
        ("clip1.MP4", "mlx-community/whisper-tiny-mlx"),
        ("clip1.MP4", "mlx-community/whisper-medium-mlx"),
        ("clip2.mp4", "mlx-community/whisper-tiny-mlx"),
        ("clip2.mp4", "mlx-community/whisper-medium-mlx"),
    ]


def test_parse_benchmark_args_supports_source_dir_and_model_overrides() -> None:
    """Verify benchmark CLI arguments override the default folder and model list."""
    source_dir, models = video_notes.parse_benchmark_args(
        ["custom-test-dir", "model-a", "model-b"]
    )

    assert source_dir == Path("custom-test-dir")
    assert models == ("model-a", "model-b")


def test_iter_mp4_files_matches_case_insensitive_extensions(tmp_path: Path) -> None:
    """Verify MP4 discovery is case-insensitive."""
    (tmp_path / "a.MP4").write_bytes(b"video")
    (tmp_path / "b.mp4").write_bytes(b"video")
    (tmp_path / "c.Mov").write_bytes(b"video")

    matched = video_notes.iter_mp4_files(tmp_path)

    assert [path.name for path in matched] == ["a.MP4", "b.mp4"]


def test_build_center_text_layout_uses_larger_font_for_shorter_text() -> None:
    """Verify shorter text layouts can use larger fonts than longer text."""
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
    """Verify the configured text scale factor affects the chosen font size."""
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
    """Verify font discovery prefers configured, system, then cached fonts."""
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
    """Verify the default note font download is written to the target path."""
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
