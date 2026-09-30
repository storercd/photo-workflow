# Crop Window Selection Priority

This document describes how left-click selection is resolved inside the image window. Toolbar controls, including the angle slider and Auto/Show actions, use their normal Qt widget hit testing and are independent of image-window priority.

## Left-Click Priority

1. **Loading state:** While the selected preview is loading, image-window clicks do not start crop gestures or select guides.
2. **Auto-angle guide:** A detected candidate line within 10 screen pixels of the pointer is selected first. Its angle is applied through the rotation slider, and the matching candidate is highlighted. This takes priority over crop handles and crop creation when they overlap.
3. **Crop plane:** If no guide was hit, clicks outside the rotated image's axis-aligned crop plane are ignored.
4. **New crop:** If there is no active crop, or the current crop covers the full image, clicking within the crop plane starts drawing a crop.
5. **Crop edge or corner:** For a partial crop, the 24-pixel edge hit zones are checked. A nearby corner selects its corner grip; otherwise a nearby edge selects that edge for resizing.
6. **Crop interior:** A click within the crop but outside all edge hit zones moves the crop.
7. **Empty crop-plane area:** Clicking outside the partial crop starts drawing a replacement crop.

## Overlapping Targets

- Horizon guides have priority over crop edges, corners, and interior movement. To adjust a crop where a guide crosses it, click away from the guide within the crop's handle or interior area.
- If two guide hit zones overlap, the nearest line is selected. If distances are exactly equal, the later candidate in the current ranked list wins.
- A guide is selectable along its displayed segment; clicking beyond its endpoints does not select it.
- Hovering within a guide's hit zone shows a pointing-hand cursor. Other crop cursor behavior remains unchanged.

## Future Changes

When adding another selectable image overlay, update this priority list and its overlap rules alongside the hit-testing implementation and tests. Keep the documented order aligned with `CropView.mousePressEvent` and the focused crop-view tests.
