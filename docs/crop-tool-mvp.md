# Rapid Crop Tool: MVP

## Purpose

Build a small macOS desktop app for quickly reviewing a folder of RAW photos, adjusting each photo's crop, and saving the crop to a Lightroom-compatible XMP sidecar. It is a crop-pass tool, not a RAW editor or photo catalog.

The first release should prove two things: the folder can be navigated without waiting for RAW development, and Lightroom interprets the crop metadata as intended.

## MVP Decisions

- **Platform/UI:** macOS desktop app using Python and PySide6/Qt.
- **Input:** One folder at a time; initially support Canon CR3 files, matching XMP sidecars, and the current Canon camera's embedded JPEG preview. Keep file discovery isolated so additional RAW formats can be added later.
- **Preview:** Extract and display an embedded JPEG preview. Decode/extract work runs off the UI thread. Show the best available preview immediately, then replace it only if a better preview becomes available. Do not demosaic RAW files in the MVP.
- **Preview memory:** Downsample decoded previews to at most 2560 pixels on the longest side and keep a 12-image LRU cache; tune these limits from measured use.
- **Persistence:** Read and write Adobe Camera Raw `crs:CropTop`, `CropLeft`, `CropBottom`, `CropRight`, and `CropAngle` as normalized crop metadata. Preserve all unrelated XMP metadata. Use ExifTool for the initial implementation, and validate the exact result in Lightroom Classic before treating the format as settled.
- **Crop flag:** Set `crs:HasCrop=True` when writing crop bounds, matching Lightroom-authored sidecars.
- **Safety:** Do not modify RAW files. Write sidecars through a serialized background writer. Flush the current photo's pending change before changing photos and before orderly app close. Report write failures visibly and do not silently discard pending changes.

## MVP User Workflow

1. Open a folder through a folder chooser or drop a folder onto the app.
2. Drop a supported RAW photo onto the app to open its folder at that photo.
3. See an ordered list/count of supported RAW files and the selected photo's preview.
4. Move to the previous/next photo with keyboard shortcuts; buttons are available as a fallback.
5. Draw or adjust a crop with the mouse.
6. Choose freeform or a locked aspect ratio. In freeform mode, optional snapping uses the configured list of acceptable ratios.
7. See the saved/pending state. Crop edits are debounced while dragging; pending changes begin writing when interaction settles, and are flushed before navigation or close.
8. Close and inspect the edited files in Lightroom Classic.

## Crop Interaction

- Display the full preview with a crop overlay.
- Drag inside the crop to reposition it; drag an edge or corner to resize it; drag outside the crop to start a new crop.
- Keep the crop within image bounds.
- Use a generous 24-pixel edge/corner hit zone for crop resizing.
- Support three modes:
  - **Freeform:** Resize without an aspect constraint; snap to a configured ratio when close enough.
  - **Locked ratio:** Resize while maintaining the selected ratio.
  - **Snap toggle:** Enable or disable snapping independently from the selected resize mode.
- Provide a small, editable ratio list in a simple settings file. Start with `1:1`, `4:5`, `5:4`, `3:2`, `2:3`, `4:3`, `3:4`, and `16:9`.
- Make the current mode and ratio visible. Choose and document a compact default keyboard map before implementation; it must include previous/next, freeform/lock toggle, snap toggle, and ratio selection.
- Initial keyboard map: `Left`/`Right` previous/next, `L` toggle ratio lock, `S` toggle snapping, `1`-`8` choose the corresponding configured ratio, `Command+O` open folder, and `Command+S` save now.
- Display the crop's current preset when it matches within 0.5%; otherwise show `Free` with its measured ratio. Lock mode captures the exact current crop ratio, including nonstandard ratios.
- Selecting a preset immediately refits the crop within its current bounds, centered on its current center. Selecting `Free` retains the existing crop geometry.
- Keep crop geometry and snapping calculations independent of Qt widgets so they can be tested without launching the UI.

## Performance Requirements

- Image extraction/decoding must never block painting or keyboard input.
- On folder open, show a placeholder and start loading the selected photo immediately.
- Keep the selected photo first in a bounded prefetch window of 8 photos in the current navigation direction (including the current photo) and 4 photos in the reverse direction. Rebuild pending priorities when navigation changes direction.
- Keep no more than three preview loads active at a time and retain at most 12 decoded previews in an LRU cache. Cancel or ignore stale results when the selection changes; tune these limits from measured use.
- Prefer the embedded JPEG over RAW development for the initial display. Keep the preview pipeline replaceable so faster or higher-quality decoders can be evaluated later.
- Measure navigation latency on a representative folder. The initial usability target is that a prefetched next/previous photo appears within one UI frame; a cache miss must keep the interface responsive and show loading feedback.
- Avoid loading full-resolution RAW pixels unless a later feature demonstrates that the embedded preview is insufficient.

## Explicitly Out of Scope for MVP

- Manual image rotation and horizon auto-adjustment.
- Side-by-side full-image/cropped-image preview.
- RAW rendering, exposure/color controls, export, catalog/database, ratings, keywords, or editing features beyond crop metadata.
- Windows/Linux support and non-CR3 formats.

These are candidates for follow-up after crop interaction, speed, and Lightroom round-trip behavior are validated. Manual rotation should be considered before automatic horizon detection.

The current prototype applies EXIF orientation to embedded previews and uses display-oriented dimensions for crop geometry. Existing Lightroom `CropAngle` metadata is preserved when saving crops; changing rotation is not yet supported.

## Reliability and XMP Compatibility

- If a matching XMP exists, load its crop and preserve its other metadata.
- If no sidecar exists, initialize a valid crop state and create the sidecar on the first edit.
- Keep crop coordinates normalized to the image dimensions and define the application's internal convention as `left, top, right, bottom` with bounds in `[0, 1]`.
- Confirm behavior for uncropped files, existing crops, rotated/oriented images, and sidecar creation with sample files.
- Prevent overlapping writes to the same sidecar. A write failure must leave the photo marked dirty and provide a retry/error path.
- Verify a changed sidecar in Lightroom Classic, save it from Lightroom, then reopen it in the app to confirm the crop round-trips without drift or loss of unrelated metadata.
- Keep a recoverable copy of the original sidecar during early development. Decide later whether to retain ExifTool's `_original` backups or replace that behavior with an explicit app backup policy.

## MVP Acceptance Criteria

- The app opens a folder containing CR3 files and associates each with the correctly named XMP sidecar.
- Previous/next navigation does not block the UI while a preview is loaded or extracted.
- A centered crop can be created, repositioned, resized, and constrained to a chosen ratio.
- Freeform snapping can be enabled/disabled, and the ratio list can be changed without editing UI code.
- Existing crop values are shown correctly when revisiting a photo.
- Crop edits are written before navigation completes and before a normal close completes; write errors remain visible and recoverable.
- The RAW file remains byte-for-byte unchanged.
- Lightroom Classic displays the expected crop after the app writes the XMP, and the app can read the crop after Lightroom saves the sidecar.
- Crop geometry, ratio locking, snapping thresholds, and boundary constraints have automated unit tests.

## Suggested Build Order

1. **Metadata proof:** Read/write a crop on representative CR3/XMP pairs; round-trip through Lightroom Classic. Keep the existing command-line experiment as a reference during this phase.
2. **Crop model:** Implement and test normalized crop geometry, ratio lock, snapping, and boundary behavior.
3. **Preview/navigation shell:** Open one folder, extract embedded previews asynchronously, display a photo, and navigate with keys.
4. **Crop interaction:** Add overlay editing and connect it to the crop model.
5. **Persistence and failure states:** Add debounced writes, serialized flush on navigation/close, dirty indication, and visible errors.
6. **Performance pass:** Measure on a representative folder; tune prefetch, cache size, and preview selection based on observed latency.
7. **Lightroom validation:** Complete the round-trip checklist and only then use the app for a larger real folder.

## Questions to Resolve During the First Prototype

- Which ExifTool embedded-preview tag gives the best speed/quality trade-off for these CR3 files?
- Does Lightroom Classic honor the calculated crop bounds directly for all tested orientations, or is additional orientation handling required?
- What snap threshold feels natural during mouse resizing? Keep it configurable until user testing settles it.
- Which keyboard bindings are comfortable and do not conflict with macOS or Qt defaults?
- What is the smallest useful preview resolution for crop composition on the target display?

## Follow-Up TODOs

- [ ] Add manual image rotation and let users adjust the horizon angle.
- [ ] Explore automatic horizon leveling after manual rotation is supported.
- [ ] Add an optional side-by-side view with the full image and crop boundaries on one side and the cropped preview on the other.
- [ ] Experiment with a smoother Lightroom Classic refresh workflow so users do not need to manually force metadata rereading after an XMP update.
- [ ] Measure navigation latency on a representative larger folder and tune preview extraction, prefetch, and cache limits based on the results.
- [ ] Add image pan and zoom controls (move and grow/shrink the crop window, not literally panning and zooming the view) without conflicting with crop creation or adjustment gestures.
