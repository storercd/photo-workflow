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
  *.CR3 / *.CR2            RAW files copied straight into live/ are ingested like uploads
  2026/10/20261031/         date folder (same naming as memory-card import)
    input/                  renamed originals (+ .xmp sidecars)
    denoise/  mask/  lighting/  output/  export/    per-stage outputs
```

Renaming is the first thing that happens to a new file (`live/ingest.py` reuses
`memory_card_copy.build_capture_dir/build_target_filename`); an existing target is
overwritten. Clean wipes the whole Live folder, input included.

## Stages

`FTP → Denoise → Mask → Lighting/Color → AutoCrop → Export → CloudSync`
(`photo_workflow.live.stages.StageId`). Every stage has a toggle. All are off by default except AutoCrop, which is on unless
the user switches it off to save processor time (a disabled AutoCrop passes through, so Export
and viewing use whatever crop already exists).
A stage starts on an image as soon as the image is eligible. Its input is the most
advanced earlier output that exists (disabled/missing/skipped stages are passed over, down
to the original), so later stages and the viewer always have something to use.
Export and CloudSync are *blocking*: while they are off, images wait there. Other stages
pass images through when off, but every image that has reached an off feature still counts
in that feature's queue as well as the one it is really at. The counts of off features blink
in the status panel, showing the work that turning them on would add.

Regenerating a stage's output (`pipeline.invalidate`) resets every downstream stage for
that image and re-runs them immediately; stale in-flight results are discarded. The photo
open in the editor is always served first (`pipeline.prioritize`), so regenerating it
never waits behind the rest of the queue.

## Failure model

A raised exception is retried with a short backoff (0.5 s, 1.5 s, 4 s). If it still
fails (or `PermanentStageError` is raised) the image is marked failed (grayed with a large
warning in the editor) and the **feature halts** with a red exclamation. Toggling the
feature off clears the error; toggling it on again retries the failed images. Everything
is logged to `~/Library/Logs/photo-workflow/live.log`. Non-fatal problems use
`job.warn(...)` / `host.warn(...)` and appear in the warning list (⚠ button).

## Idle Processing

An option (**off by default**) that only processes while you are idle. Switch it on with
the **Idle Processing** toggle button in the Live status panel or the checkbox in
Preferences → Live (they stay in sync and both persist `idle_processing`); the delay is set in
Preferences. The name is the single constant `IDLE_PROCESSING_LABEL` in `live/settings.py`. While the option is on, `IdleMonitor`
(`editor/live_ui.py`) holds the pipeline (`LivePipeline.set_held`) whenever the mouse
moved, or a click, key press or scroll happened, within the idle delay (default 3 s,
`idle_delay_seconds`). Once everything has been still for that long the hold is released
and waiting work starts at once. A hold only stops *new* jobs from starting, so running
jobs finish; arriving files are still ingested and queued. The status panel shows an italic
"Waiting for idle" line while held (counts do not blink for this).

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

## Preprocessing features

Three stage modules live in `live/features/`: `denoise.py`, `mask.py`, `lighting.py`.
Intermediates are 16-bit scene-linear sRGB TIFFs (`live/imaging.py`; RAW is developed with
LibRaw via `rawpy`: camera white balance, no tone curve, cropped to the active area).

- **Denoise** → `denoise/<name>.tif`: non-local-means on a luma/chroma split in a
  square-root domain, with the filter strength scaled to the noise measured in each image
  (so high-ISO frames are cleaned properly and clean frames are left nearly untouched).
  Chroma is smoothed at quarter resolution because colour noise is blotchy, not grainy.
  There is no library that re-encodes a mosaiced CR2/CR3, so the "raw-format" intermediate
  is the linear developed image.
- **Mask** → `mask/<name>.tif`: Apple Vision (`VNGenerateForegroundInstanceMaskRequest`,
  macOS only, `pyobjc-framework-Vision`) finds the main subject; everything outside it is
  scaled by -0.25 stop in linear light. No subject is an expected case, not a
  warning: the stage is `StageResult.skipped` (later stages use the earlier source) and the
  viewer shows a small amber "Mask not applied: no subject detected" badge (top right) for that
  photo while Mask is enabled. Real mask errors are still failures.
- **Camera look** (`imaging.match_camera_look`): linear data has no tone curve or picture
  style, so shown plainly it is dark and flat next to the camera's own JPEG. Wherever a
  RAW or linear intermediate becomes a display image (Lighting/Color input, Export without
  Lighting), a monotonic per-channel curve is fitted from it to the RAW's embedded JPEG
  and applied at full size, so the starting point matches the camera's rendering. It falls
  back to the plain sRGB encoding if the embedded preview is missing or a different shape.
- **Lighting/Color** → `output/<name>.jpg` (quality 95, 4:4:4, sRGB profile): reuses
  `editor/lighting.py` (`estimate_adjustment`/`apply_adjustment`/`gains_from_kelvin`). Settings:
  `auto_lighting` (default on) and `color_temperature` (Kelvin override, default none).

`live/detection.py` is the shared detection API: Mask publishes normalized subject (and ball)
boxes per photo; Automatic Cropping calls `get_detection(photo)` or
`detect_photo(photo, source)` (detects and caches when missing).
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
- **Viewer indicator:** in Live mode the crop frame is blue with an "Auto crop" label while
  the sidecar is marked `auto`, and white with "Manual crop" otherwise. Any edit (crop or
  angle) saves as manual, so the frame flips to manual at once.
- **Detection:** the stage uses a `Detector`. The default (`SharedDetector`) reads
  `live.detection.get_detection(photo)` (cached by Mask) or runs `detect_photo(photo, source)`
  on the best source, and converts the normalized boxes to pixels of the photo's displayed
  size with `BoundingBox.to_pixels`. The ball is only used with the ball option; direction is
  unknown (0). With no subject, or no detection backend, the stage skips with a warning and
  leaves the sidecar untouched.

## Code map

| Module | Role |
| --- | --- |
| `live/stages.py` | stage ids, order, specs, subfolders |
| `live/pipeline.py` | Qt-free framework (state, invalidation, retries, snapshot) |
| `live/api.py` | extension interface above |
| `live/settings.py` | Live settings; secrets in keychain (`keyring`) |
| `editor/live_ui.py` | controller, status panel, hide/show tab, Clean confirmation |
| `editor/app.py` | Live mode toggle, Live photo list, failure overlay |

Finished stage progress is saved to `<live folder>/.pipeline-state.json` (`live/persistence.py`)
and restored at launch, so a restart does not redo finished work. A stage is redone when its
output file is missing, the original changed, its feature was toggled while the app was closed
(Denoise, Mask, Lighting/Color), or the AutoCrop options changed.

## FTP feature

`live/features/ftp.py` runs a `pyftpdlib` server while the FTP feature is on (off by
default; set the username, password and port in Preferences → Live). Point the camera at
the machine's LAN address and the configured port, passive mode, with the same login.
Uploads are staged in `incoming/.ftp-uploading` and moved to `incoming` only when the
transfer completes, then passed to `host.ingest`. Only loopback, private and link-local
peers may connect; the account can upload but not delete or create folders. A dropped
connection mid-upload is discarded and shown as a warning. Passive data ports are
60000–60099. The address and port to enter on the camera are shown under the FTP row of the
status panel while the server runs (`StageImplementation.status_text`, polled through
`LivePipeline.stage_info`) and logged at start; changing credentials or the port takes effect
after toggling FTP off and on.

## Export and Cloud Sync

**Export** (`features/export.py`) renders each photo's best source (the Lighting/Color JPEG,
else the Mask or Denoise 16-bit linear TIFF encoded to sRGB, else the RAW's largest embedded
JPEG; no RAW developer is bundled) with the crop and straighten angle from its XMP sidecar
(written by AutoCrop or by hand) into a q95 JPEG in `export/`. Saving a crop in the editor
re-runs Export and Cloud Sync for that photo.

**Cloud Sync** (`features/cloud_sync.py`) follows `storercd/photo-portal`'s `sync/sync.py`:
for each exported image it uploads `photos/{thumb,med,web,full}/<stem>-<hash>.jpg` (long
edges 600/1200/2000 px plus the untouched export) to the private R2 bucket from Live
settings, skipping keys that already exist and deleting keys of older exports. The only
filter is *Cloud Sync minimum stars* (`r2_min_rating`, read from the sidecar's
`xmp:Rating`): photos below it are deleted from the bucket. Saving a rating in the editor
re-runs Cloud Sync for that photo. Incomplete R2 settings halt the feature immediately.
