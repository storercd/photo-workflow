# Live mode (developer note)

Live mode is a third editor mode next to Crop and Review. Photos arrive in a **Live
folder**, move through a staged background pipeline, and are viewed/cropped in the
editor. This note covers the framework and how a feature plugs in. The foundation ships
with **every feature off** and every stage a no-op; features are added one at a time.

## Layout

The Live folder defaults to the camera folder's path with only the last folder name
replaced by `live` (`photo_workflow.live.layout.default_live_folder`); override it in
Preferences → Live.

```text
live/
  incoming/                 landing folder for new uploads (not yet renamed)
  2026/10/20261031/         date folder (same naming as memory-card import)
    input/                  renamed originals (+ .xmp sidecars)
    denoise/  mask/  lighting/  output/  export/    per-stage outputs
```

Renaming is the first thing that happens to a new file (`live/ingest.py` reuses
`memory_card_copy.build_capture_dir/build_target_filename`); an existing target is
overwritten. Clean wipes the whole Live folder, input included.

## Stages

`FTP → Denoise → Mask → Lighting/Color → AutoCrop → Export → CloudSync`
(`photo_workflow.live.stages.StageId`). AutoCrop is always on; the rest have a toggle.
A stage starts on an image as soon as the image is eligible. Its input is the most
advanced earlier output that exists (disabled/missing/skipped stages are passed over, down
to the original), so later stages and the viewer always have something to use.
Export and CloudSync are *blocking*: while they are off, images wait there and their
count blinks in the status panel. Other stages pass images through when off.

Regenerating a stage's output (`pipeline.invalidate`) resets every downstream stage for
that image and re-runs them immediately; stale in-flight results are discarded.

## Failure model

A raised exception is retried with a short backoff (0.5 s, 1.5 s, 4 s). If it still
fails (or `PermanentStageError` is raised) the image is marked failed (grayed with a large
warning in the editor) and the **feature halts** with a red exclamation. Toggling the
feature off clears the error; toggling it on again retries the failed images. Everything
is logged to `~/Library/Logs/photo-workflow/live.log`. Non-fatal problems use
`job.warn(...)` / `host.warn(...)` and appear in the warning list (⚠ button).

## Writing a feature

Add one file, `src/photo_workflow/live/features/<name>.py`. Every non-underscore module in
that package is imported at start-up; nothing else needs editing.

```python
from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.stages import StageId


class Denoise(StageImplementation):
    def process(self, job: StageJob) -> StageResult:
        out = job.output_path(suffix=".dng")   # live/.../denoise/<name>.dng, folder created
        ...                                    # read job.source, write out (overwrite is fine)
        return StageResult.done(out)           # last output is what later stages/viewer use

    # Optional, for features that run a service while enabled (e.g. the FTP server):
    def start(self, host): ...                 # host.ingest(path) hands over an upload
    def stop(self): ...


def register(registry: StageRegistry) -> None:
    registry.register(StageId.DENOISE, Denoise())
```

Rules: `process` runs on a worker thread (never touch Qt); read settings from
`job.settings` (credentials come from the system keychain); return
`StageResult.skipped()` when there is nothing to do; raise to fail. Lighting writes its
display-oriented JPEG to `output/` (`job.output_path("output", ".jpg")`); Export output is
never used for viewing. AutoCrop writes the crop into the photo's XMP sidecar.

## AutoCrop

`live/features/autocrop.py` computes a crop per photo and writes it to the XMP sidecar via
`editor/metadata.write_photo_crop(..., source="auto")`; scoring is the fixed, Qt-free
`live/autocrop_scoring.py` (spec: lead room, size, ball, thirds, edge tension, headroom).
Only ratios from Preferences → Live are used. Options: *include the ball* and *Force
automatic crop*; changing either (or the ratios) re-runs AutoCrop on every photo.

- **Manual vs auto:** sidecars carry `pwf:CropSource` (`manual`/`auto`, namespace in
  `editor/resources/photo_workflow_xmp.config`). `write_photo_crop` defaults to `manual`, so
  Crop mode needs no changes. A crop without a marker predates auto crop and counts as manual.
  `is_manual_crop` is what the stage checks; it always computes, but only overwrites a manual
  crop when Force is on.
- **Detection:** the stage uses a `Detector` (`detect(image, want_ball=) -> Detection | None`).
  The default imports `photo_workflow.live.detection.detect_scene(image, want_ball=...)`
  returning an object with `image_width`, `image_height`, `subject`, `ball`, `direction`
  (boxes `(x0, y0, x1, y1)` in displayed orientation). If that is missing the stage logs once
  per photo and skips with a warning, leaving the sidecar untouched.

## Code map

| Module | Role |
| --- | --- |
| `live/stages.py` | stage ids, order, specs, subfolders |
| `live/pipeline.py` | Qt-free framework (state, invalidation, retries, snapshot) |
| `live/api.py` | extension interface above |
| `live/settings.py` | Live settings; secrets in keychain (`keyring`) |
| `editor/live_ui.py` | controller, status panel, hide/show tab, Clean confirmation |
| `editor/app.py` | Live mode toggle, Live photo list, failure overlay |

Pipeline state is held in memory only; after a restart it is re-derived from the folder and
work re-runs.
