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

1. Detect a mounted memory card, copy its files to a temporary staging directory
	on the target volume, read their capture timestamps, move them into capture-date
	folders, verify the import, delete the copied files from the card, eject the card,
	and report remaining disk space.
2. Scan the dated camera folder for `.mp4` files. Short clips produce `.tif`
	transcription notes; longer clips produce `.tif` reminder cards containing
	five evenly spaced video frames. Generated cards are placed with the photos,
	and the videos are moved into the dated folder's `videos` subfolder.
3. Assess any `_Rejected` folders under the configured camera root, report the
	reclaimable disk space for each folder and the total, and optionally purge
	them.
4. Open the processed dated camera folder in Finder after all steps complete.

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
Imported filenames use `YYYYMMDD_HHMMSS_FF_original-name.ext`, where `FF` is
hundredths of a second. Capture time comes from `DateTimeOriginal`, `CreateDate`,
or `MediaCreateDate`, in that order. Files without usable metadata fall back to
their preserved filesystem modification time and produce a warning. A source
folder name is added only if two complete destination filenames still collide.

Staging directories are hidden under `camera_root` so final moves stay on the
same volume. They are removed after the run, including when an import fails.

Run it with:

```bash
uv run photo-workflow-run
```

Add `--purge-rejected` to delete the assessed `_Rejected` folders after the
report is logged:

```bash
uv run photo-workflow-run --purge-rejected
```

Remove unused `.jpg` camera companion files beneath a root folder with:

```bash
uv run photo-workflow-purge-jpegs /path/to/photos
```

The command reports the number of files and disk space recovered. Add
`--dry-run` to report what would be removed without deleting files. When the
root is omitted, it uses the configured `workflow.camera_root`.

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

This workflow requires `exiftool`, `ffmpeg`, and `ffprobe` to be available on `PATH`.

## Rapid Crop Tool

Launch the folder-oriented crop editor with an optional starting folder:

```bash
uv run photo-workflow-crop /path/to/photo/folder
```

Folders and individual `.CR3` photos can also be dropped onto the app window. A
dropped photo opens its parent folder with that photo selected.

The app currently supports Canon `.CR3` files and uses their embedded JPEG
previews, loading a small preview before the larger `JpgFromRaw` preview.
Previous/next are `Left`/`Right`; `L` toggles ratio lock, `S` toggles snapping,
`1`-`8` select a configured ratio, `Command+O` opens a folder, and `Command+S`
saves the current crop. Crop changes are written to the matching XMP sidecar;
RAW files are not modified. The app requires ExifTool on `PATH`.

Configure ratios and the relative snap tolerance in `photo-workflow.toml`:

```toml
[crop_tool]
aspect_ratios = ["1:1", "4:5", "5:4", "3:2", "2:3", "4:3", "3:4", "16:9"]
snap_tolerance = 0.025
```

See [docs/crop-tool-mvp.md](docs/crop-tool-mvp.md) for the MVP scope and
validation checklist. Lightroom round-trip validation is still required before
using the prototype on a large folder.

## Fonts

The note renderer resolves fonts in this order:

1. `PHOTO_WORKFLOW_FONT_PATH`, if set
2. System font fallbacks
3. A cached downloaded font installed by this project

Install the default cached font once with:

```bash
uv run photo-workflow-install-font
```
