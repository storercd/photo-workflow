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

## First Workflow

The first workflow scans the dated camera folder for short `.mp4` files, runs
local transcription, and creates a `.tif` note image with a red frame and the
detected speech centered on the canvas.

By default it uses today's folder under:

```bash
/Users/christopherstorer/working/camera/YYYYMMDD
```

Run it with:

```bash
uv run photo-workflow-short-video-notes
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
