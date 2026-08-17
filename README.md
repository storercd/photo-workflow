# photo-workflow

Daily photo processing workflows.

## Development

Create a Python 3.13 environment, then install the package and development tools:

```bash
python3.13 -m pip install -e .[dev]
```

Run checks:

```bash
ruff check .
pytest
```

## Workflow

The workflow now runs in three steps:

1. Detect a mounted memory card, flatten-copy its files into the dated camera
	folder, verify the copy, delete the copied files from the card, eject the
	card, and report remaining disk space.
2. Scan the dated camera folder for short `.mp4` files, run local
	transcription, and create `.tif` note images with a red frame and centered
	speech text.
3. Assess any `_Rejected` folders under the configured camera root, report the
	reclaimable disk space for each folder and the total, and optionally purge
	them.

By default it uses today's folder under:

```bash
/Users/christopherstorer/working/camera/YYYYMMDD
```

You can override that base folder with [photo-workflow.toml](photo-workflow.toml):

```toml
[workflow]
camera_root = "/Users/christopherstorer/working/camera"

[memory_card_copy]
card_mount_root = "/Volumes"
low_disk_warning_gb = 30.0
low_disk_warning_percent = 5.0
copy_verification = "basic"
halt_on_insufficient_space = true
ignored_extensions = [".ctg", ".log", ".tmp"]

[video_notes]
max_duration_seconds = 10.0
transcription_model = "mlx-community/whisper-medium-mlx"
```

`copy_verification = "basic"` checks file existence and size after copy. Set it
to `"crc32"` to read both source and destination and compare CRC32 checksums.
`ignored_extensions` skips known non-media sidecar files during copy and delete.
`halt_on_insufficient_space = true` stops an import before copying when the target
volume has fewer free bytes than the planned source files require.

Run it with:

```bash
uv run photo-workflow-run
```

Add `--purge-rejected` to delete the assessed `_Rejected` folders after the
report is logged:

```bash
uv run photo-workflow-run --purge-rejected
```

Run the steps individually with:

```bash
uv run photo-workflow-memory-card-copy
uv run photo-workflow-short-video-notes
uv run photo-workflow-benchmark-transcription
```

The benchmark command defaults to `video-test/` and compares `tiny`, `small`,
and `medium` MLX Whisper models. You can override both the folder and models:

```bash
uv run photo-workflow-benchmark-transcription video-test \
	mlx-community/whisper-tiny-mlx \
	mlx-community/whisper-small-mlx \
	mlx-community/whisper-medium-mlx
```

This workflow requires `ffmpeg` and `ffprobe` to be available on `PATH`.

## Fonts

The note renderer resolves fonts in this order:

1. `PHOTO_WORKFLOW_FONT_PATH`, if set
2. System font fallbacks
3. A cached downloaded font installed by this project

Install the default cached font once with:

```bash
uv run photo-workflow-install-font
```
