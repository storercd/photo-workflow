"""Workflow for turning short videos into note images."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper
from PIL import Image, ImageDraw, ImageFont

from photo_workflow.config import VideoNotesConfig, build_today_source_dir, load_video_notes_config

DEFAULT_OUTPUT_EXTENSION = ".tif"
DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-tiny"
DEFAULT_TEXT_SCALE_FACTOR = 0.5
DEFAULT_FONT_CACHE_ENV_VAR = "PHOTO_WORKFLOW_FONT_CACHE_DIR"
DEFAULT_FONT_PATH_ENV_VAR = "PHOTO_WORKFLOW_FONT_PATH"
DEFAULT_DOWNLOADED_FONT_FILENAME = "Inter[opsz,wght].ttf"
DEFAULT_DOWNLOADED_FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/"
    "ofl/inter/Inter%5Bopsz,wght%5D.ttf"
)
DEFAULT_NOTE_WIDTH = 4032
DEFAULT_NOTE_HEIGHT = 3024
DEFAULT_NOTE_BORDER_WIDTH = 80
DEFAULT_TEXT_MARGIN_MULTIPLIER = 6
DEFAULT_MAX_FONT_SIZE = 600
DEFAULT_MIN_FONT_SIZE = 120
DEFAULT_FONT_SIZE_STEP = 20
DEFAULT_MIN_LINE_SPACING = 24
DEFAULT_LINE_SPACING_DIVISOR = 6
DEFAULT_FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedVideoNote:
    """Result for a video that produced a note image."""

    video_path: Path
    output_path: Path
    duration_seconds: float
    transcription: str


def process_short_videos(
    source_dir: Path | None = None,
    *,
    max_duration_seconds: float | None = None,
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
) -> list[ProcessedVideoNote]:
    """Process short MP4 clips in the dated source directory."""

    active_source_dir = source_dir or build_today_source_dir()
    active_max_duration_seconds = (
        load_video_notes_config().max_duration_seconds
        if max_duration_seconds is None
        else max_duration_seconds
    )
    if not active_source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {active_source_dir}")

    processed_notes: list[ProcessedVideoNote] = []
    for video_path in iter_mp4_files(active_source_dir):
        duration_seconds = probe_video_duration_seconds(video_path)
        if duration_seconds >= active_max_duration_seconds:
            continue

        transcription = transcribe_video(video_path, model=transcription_model)
        output_path = video_path.with_suffix(DEFAULT_OUTPUT_EXTENSION)
        create_note_image(output_path, transcription)
        copy_file_timestamp(video_path, output_path)

        processed_notes.append(
            ProcessedVideoNote(
                video_path=video_path,
                output_path=output_path,
                duration_seconds=duration_seconds,
                transcription=transcription,
            )
        )

    return processed_notes


def iter_mp4_files(source_dir: Path) -> list[Path]:
    """Return MP4 files in the source directory, regardless of extension case."""

    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def probe_video_duration_seconds(video_path: Path) -> float:
    """Return the clip duration using ffprobe."""

    ffprobe_path = require_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


def transcribe_video(video_path: Path, *, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> str:
    """Extract mono audio and return a normalized transcription."""

    ffmpeg_path = require_tool("ffmpeg")
    with tempfile.TemporaryDirectory(prefix="photo-workflow-") as tmp_dir:
        audio_path = Path(tmp_dir) / f"{video_path.stem}.wav"
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model, verbose=False)

    text = " ".join((result.get("text") or "").split())
    return text or "[No speech detected]"


def create_note_image(output_path: Path, transcription: str) -> None:
    """Render a note image with a red frame and centered transcription."""

    width = DEFAULT_NOTE_WIDTH
    height = DEFAULT_NOTE_HEIGHT
    border_width = DEFAULT_NOTE_BORDER_WIDTH
    font, wrapped_text, line_spacing = build_center_text_layout(
        transcription=transcription,
        image_size=(width, height),
        border_width=border_width,
    )

    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline="red", width=border_width)

    text_box = draw.multiline_textbbox(
        (0, 0),
        wrapped_text,
        font=font,
        spacing=line_spacing,
        align="center",
    )
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_position = ((width - text_width) / 2, (height - text_height) / 2)
    draw.multiline_text(
        text_position,
        wrapped_text,
        fill="white",
        font=font,
        spacing=line_spacing,
        align="center",
    )

    image.save(output_path, format="TIFF", compression="tiff_lzw")


def build_center_text_layout(
    transcription: str,
    *,
    image_size: tuple[int, int],
    border_width: int,
) -> tuple[ImageFont.ImageFont | ImageFont.FreeTypeFont, str, int]:
    """Find the largest fitting layout, then render at a scaled-down size."""

    width, height = image_size
    max_text_width = width - (border_width * DEFAULT_TEXT_MARGIN_MULTIPLIER)
    max_text_height = height - (border_width * DEFAULT_TEXT_MARGIN_MULTIPLIER)
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    for font_size in range(
        DEFAULT_MAX_FONT_SIZE,
        DEFAULT_MIN_FONT_SIZE - 1,
        -DEFAULT_FONT_SIZE_STEP,
    ):
        font = load_note_font(font_size)
        line_spacing = max(DEFAULT_MIN_LINE_SPACING, font_size // DEFAULT_LINE_SPACING_DIVISOR)
        wrapped_text = wrap_text_to_pixel_width(
            transcription,
            font=font,
            max_text_width=max_text_width,
            draw=measure_draw,
        )
        bbox = measure_draw.multiline_textbbox(
            (0, 0),
            wrapped_text,
            font=font,
            spacing=line_spacing,
            align="center",
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= max_text_width and text_height <= max_text_height:
            scaled_font_size = max(
                DEFAULT_MIN_FONT_SIZE,
                int(font_size * DEFAULT_TEXT_SCALE_FACTOR),
            )
            scaled_font = load_note_font(scaled_font_size)
            scaled_line_spacing = max(
                DEFAULT_MIN_LINE_SPACING,
                scaled_font_size // DEFAULT_LINE_SPACING_DIVISOR,
            )
            scaled_wrapped_text = wrap_text_to_pixel_width(
                transcription,
                font=scaled_font,
                max_text_width=max_text_width,
                draw=measure_draw,
            )
            return scaled_font, scaled_wrapped_text, scaled_line_spacing

    fallback_font = load_note_font(DEFAULT_MIN_FONT_SIZE)
    fallback_spacing = max(
        DEFAULT_MIN_LINE_SPACING,
        DEFAULT_MIN_FONT_SIZE // DEFAULT_LINE_SPACING_DIVISOR,
    )
    fallback_text = wrap_text_to_pixel_width(
        transcription,
        font=fallback_font,
        max_text_width=max_text_width,
        draw=measure_draw,
    )
    return fallback_font, fallback_text, fallback_spacing


def wrap_text_to_pixel_width(
    text: str,
    *,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_text_width: int,
    draw: ImageDraw.ImageDraw,
) -> str:
    """Wrap text by measured pixel width instead of character count."""

    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        candidate_line = f"{current_line} {word}"
        if draw.textlength(candidate_line, font=font) <= max_text_width:
            current_line = candidate_line
            continue

        lines.append(current_line)
        current_line = word

    lines.append(current_line)
    return "\n".join(lines)


def load_note_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a large scalable font for note rendering."""

    for font_path in iter_note_font_paths():
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            continue

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def iter_note_font_paths() -> list[Path]:
    """Return local font candidates in preference order."""

    font_paths: list[Path] = []
    configured_font_path = os.environ.get(DEFAULT_FONT_PATH_ENV_VAR)
    if configured_font_path:
        font_paths.append(Path(configured_font_path).expanduser())

    font_paths.extend(Path(font_path) for font_path in DEFAULT_FONT_PATHS)

    cached_font_path = build_cached_font_path()
    if cached_font_path.exists():
        font_paths.append(cached_font_path)

    return font_paths


def build_font_cache_dir() -> Path:
    """Return the local cache directory for downloaded fonts."""

    configured_cache_dir = os.environ.get(DEFAULT_FONT_CACHE_ENV_VAR)
    if configured_cache_dir:
        return Path(configured_cache_dir).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "photo-workflow" / "fonts"

    return Path.home() / ".cache" / "photo-workflow" / "fonts"


def build_cached_font_path() -> Path:
    """Return the path of the project's cached default font."""

    return build_font_cache_dir() / DEFAULT_DOWNLOADED_FONT_FILENAME


def install_default_note_font(
    destination: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download and cache the default note font once for later runs."""

    target_path = destination or build_cached_font_path()
    if target_path.exists() and not force:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DEFAULT_DOWNLOADED_FONT_URL, timeout=30) as response:
        font_bytes = response.read()

    if not font_bytes:
        raise ValueError("Downloaded font file was empty")

    target_path.write_bytes(font_bytes)
    return target_path


def copy_file_timestamp(source_path: Path, target_path: Path) -> None:
    """Copy atime and mtime from the source file to the generated file."""

    source_stat = source_path.stat()
    os.utime(target_path, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))


def require_tool(name: str) -> str:
    """Return the full path to a required external binary."""

    tool_path = shutil.which(name)
    if tool_path is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {name}")
    return tool_path


def run_video_notes_step(source_dir: Path, *, config: VideoNotesConfig) -> None:
    """Process short videos in the dated source directory."""

    total_videos = len(iter_mp4_files(source_dir)) if source_dir.exists() else 0
    if not source_dir.exists():
        LOGGER.info("no .mp4 files found in %s", source_dir)
        return

    processed_notes = process_short_videos(
        source_dir=source_dir,
        max_duration_seconds=config.max_duration_seconds,
    )
    if not processed_notes:
        if total_videos == 0:
            LOGGER.info("no .mp4 files found in %s", source_dir)
        else:
            LOGGER.info(
                "found %s .mp4 file(s) in %s, but none were shorter than %s seconds",
                total_videos,
                source_dir,
                format_duration_seconds(config.max_duration_seconds),
            )
        return

    for note in processed_notes:
        LOGGER.info(
            "created %s from %s: %s",
            note.output_path.name,
            note.video_path.name,
            note.transcription,
        )


def main() -> None:
    """Run the short-video note workflow for today's camera folder."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_video_notes_step(build_today_source_dir(), config=load_video_notes_config())


def install_font_main() -> None:
    """Download and cache the default note font for later workflow runs."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    installed_font_path = install_default_note_font()
    LOGGER.info("installed font at %s", installed_font_path)


def format_duration_seconds(duration_seconds: float) -> str:
    """Return a stable, human-readable duration string for logging."""

    if duration_seconds.is_integer():
        return str(int(duration_seconds))

    return str(duration_seconds)


if __name__ == "__main__":
    main()
