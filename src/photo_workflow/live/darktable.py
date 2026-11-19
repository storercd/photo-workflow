"""
Profiled denoise through `darktable-cli`.

darktable's "denoise (profiled)" module uses per-camera, per-ISO noise profiles on the RAW
data, which beats any generic filter on high-ISO frames. The sidecar adds only that module and
the workflow is set to "none" (no exposure or tone mapping), so the result is the developed RAW
with camera white balance; it is exported as a 16-bit sRGB TIFF and decoded
back to the scene-linear form the other stages expect.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_workflow.live.imaging import MAX_16

CLI_NAME = "darktable-cli"
CLI_ENV = "DARKTABLE_CLI"
CLI_LOCATIONS = (
    Path("/Applications/darktable.app/Contents/MacOS/darktable-cli"),
    Path.home() / "Applications/darktable.app/Contents/MacOS/darktable-cli",
)
TIMEOUT_SECONDS = 600
# `denoiseprofile` parameters, version 12: non-local means, automatic strength.
MODE_NLMEANS_AUTO = 3
PARAMS_VERSION = 12
CURVE_NODES = 7
CURVE_CHANNELS = 6
SRGB_THRESHOLD = 0.04045


class DarktableError(RuntimeError):
    """Raised when darktable-cli is unavailable or fails."""


def find_cli() -> Path | None:
    """
    Locate `darktable-cli`.

    Returns:
        The executable named by `DARKTABLE_CLI`, else the one on `PATH` or in a standard
        macOS application folder, or `None` when darktable is not installed.
    """
    candidates = [Path(path) for path in (os.environ.get(CLI_ENV), shutil.which(CLI_NAME)) if path]
    for candidate in (*candidates, *CLI_LOCATIONS):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def denoise_params() -> bytes:
    """
    Build the `denoiseprofile` module parameters: automatic non-local-means denoise.

    Returns:
        The binary parameter block darktable stores in a sidecar history entry.
    """
    nodes = [index / (CURVE_NODES - 1) for index in range(CURVE_NODES)] * CURVE_CHANNELS
    flat = [0.5] * (CURVE_NODES * CURVE_CHANNELS)
    return struct.pack(
        f"<8f3f3fi{len(nodes)}f{len(flat)}f5i",
        1.0, 7.0, 1.2, 0.0, 0.0, 0.0, 0.1, 1.0,  # radius..overshooting
        -1.0, 0.0, 0.0,  # a: -1 selects the camera/ISO profile automatically
        0.0, 0.0, 0.0,  # b
        MODE_NLMEANS_AUTO,
        *nodes,
        *flat,
        1, 1, 1, 1, 0,  # anscombe/vst flags, wavelet colour mode, reserved
    )  # fmt: skip


def denoise_sidecar() -> str:
    """
    Build an XMP sidecar that adds profiled denoise to darktable's default pipeline.

    Returns:
        The sidecar text.
    """
    description = (
        'xmlns:darktable="http://darktable.sf.net/" darktable:xmp_version="5" '
        'darktable:raw_params="0" darktable:auto_presets_applied="0" darktable:history_end="1"'
    )
    entry = (
        'darktable:num="0" darktable:operation="denoiseprofile" darktable:enabled="1" '
        f'darktable:modversion="{PARAMS_VERSION}" darktable:params="{denoise_params().hex()}" '
        'darktable:multi_name="" darktable:multi_name_hand_edited="0" '
        'darktable:multi_priority="0"'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f"<rdf:Description {description}>\n"
        f"<darktable:history><rdf:Seq>\n<rdf:li {entry}/>\n"
        "</rdf:Seq></darktable:history></rdf:Description></rdf:RDF></x:xmpmeta>\n"
    )


def srgb_to_linear(encoded: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """
    Decode sRGB-encoded 16-bit values to linear light.

    Returns:
        A 16-bit array of the same shape.
    """
    value = encoded.astype(np.float32) / MAX_16
    linear = np.where(value <= SRGB_THRESHOLD, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    return np.round(np.clip(linear, 0, 1) * MAX_16).astype(np.uint16)


def denoise_raw(raw: Path) -> NDArray[np.uint16]:
    """
    Develop a RAW with darktable's profiled denoise.

    Returns:
        An upright `H x W x 3` linear 16-bit RGB array.

    Raises:
        DarktableError: If darktable-cli is missing, fails, or produces no readable image.
    """
    cli = find_cli()
    if cli is None:
        raise DarktableError("darktable-cli is not installed")
    with tempfile.TemporaryDirectory(prefix="photo-workflow-darktable-") as scratch:
        folder = Path(scratch)
        sidecar = folder / "denoise.xmp"
        sidecar.write_text(denoise_sidecar(), encoding="utf-8")
        output = folder / "out.tif"
        command = [
            str(cli),
            str(raw.absolute()),
            str(sidecar),
            str(output),
            "--core",
            "--library",
            ":memory:",
            "--configdir",
            str(folder / "config"),
            "--conf",
            "plugins/darkroom/workflow=none",
            "--conf",
            "plugins/imageio/format/tiff/bpp=16",
            "--conf",
            "plugins/imageio/format/tiff/compress=0",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=folder,
            )
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", "") or str(error)
            raise DarktableError(f"darktable-cli failed: {detail.strip()[-300:]}") from error
        image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.dtype != np.uint16:
            raise DarktableError("darktable-cli produced no 16-bit image")
        return srgb_to_linear(cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB))
