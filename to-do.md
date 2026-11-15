# To-do

Things deliberately deferred during the Live mode proof of concept. All Live features
(except the always-on AutoCrop) ship disabled by default; see `docs/live-mode.md`.

## Rollout

- [ ] Re-enable and verify each Live feature one at a time (FTP, Denoise, Mask, Lighting/Color,
      Export, CloudSync); leave any that misbehave disabled while troubleshooting.
- [ ] Open pull requests once the features are verified.

## Automatic crop

- [ ] Refine and tune the composition logic and its weights (currently untuned).
- [ ] Detect direction of travel (currently always 0, i.e. unknown).
- [ ] Crop angle is always 0; add straightening if wanted.
- [ ] Automatically add the complementary ratio (e.g. 2:3 for an accepted 3:2).
- [ ] Better ball detection (currently only the first Hough candidate is used).

## Image processing

- [ ] Tune Denoise strength (tested on a single sample).
- [ ] Denoise writes a linear TIFF, not a re-mosaiced RAW; decide whether that is acceptable.
- [ ] Verify Apple Vision subject masking on a range of real images.
- [ ] Non-macOS detection fallback (Vision is macOS-only; Mask and detection skip elsewhere).
- [ ] Verify rawpy development on real RAW files.

## Export

- [ ] A real RAW developer for Export instead of the embedded JPEG.
- [ ] Verify exiftool export and the crop rotation direction on real images.

## FTP

- [ ] Test with the real Canon R6 II: destination folder, passive ports (60000-60099),
      ALLO/SIZE handling, and truncated-upload detection.
- [ ] Show the FTP address/port in more places in the UI.
- [ ] Apply FTP credential/port changes without toggling the feature off and on.

## Cloud sync

- [ ] Test against real R2 credentials and bucket.
- [ ] Write a manifest (`collection.json`) to the bucket.
- [ ] Possibly build a viewing website for the synced collection.
