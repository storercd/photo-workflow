"""On-disk layout of the Live folder."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from photo_workflow.live.stages import INPUT_SUBFOLDER

LIVE_FOLDER_NAME = "live"
INCOMING_FOLDER_NAME = "incoming"
RAW_SUFFIXES = frozenset({".cr2", ".cr3"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
PARTIAL_SUFFIXES = frozenset({".part", ".tmp", ".filepart", ".xmp"})


def default_live_folder(camera_root: Path) -> Path:
    """
    Return the Live folder placed next to the camera folder.

    The path is the camera folder's path with only the final folder name replaced.

    Returns:
        The default Live folder path.
    """
    return camera_root.with_name(LIVE_FOLDER_NAME)


@dataclass(frozen=True)
class LiveLayout:
    """
    Paths inside a Live folder.

    The folder holds an `incoming` landing folder for newly arrived (not yet renamed)
    files and the usual `YYYY/MM/YYYYMMDD` date folders. Each date folder has an
    `input` subfolder with the renamed originals, plus one subfolder per stage output
    (`denoise`, `mask`, `lighting`, `output`, `export`, ...). A photo is identified by
    the path of its renamed original in `input`; its XMP sidecar sits beside it.
    """

    root: Path

    @property
    def incoming(self) -> Path:
        """Folder where uploads land before they are renamed."""
        return self.root / INCOMING_FOLDER_NAME

    def date_dir(self, photo: Path) -> Path:
        """Return the date folder holding a photo's `input` subfolder."""
        return photo.parent.parent

    def stage_dir(self, photo: Path, subfolder: str) -> Path:
        """Return a stage output subfolder of the photo's date folder."""
        return self.date_dir(photo) / subfolder

    def output_path(self, photo: Path, subfolder: str, suffix: str | None = None) -> Path:
        """
        Return where a stage should write its output for a photo.

        The file keeps the photo's stem; `suffix` (for example `.jpg`) replaces the
        extension. Existing files are expected to be overwritten.

        Returns:
            The output file path (its folder is not created).
        """
        name = photo.name if suffix is None else photo.with_suffix(suffix).name
        return self.stage_dir(photo, subfolder) / name

    def is_photo(self, path: Path) -> bool:
        """Return whether a path is a supported RAW file."""
        return path.is_file() and path.suffix.lower() in RAW_SUFFIXES

    def list_photos(self) -> list[Path]:
        """Return every renamed photo, in chronological (file name) order."""
        photos = [
            path
            for path in self.root.glob(f"*/*/*/{INPUT_SUBFOLDER}/*")
            if path.suffix.lower() in RAW_SUFFIXES
        ]
        return sorted(photos, key=lambda path: (path.name, str(path)))

    def list_incoming(self) -> list[Path]:
        """Return completed-looking files waiting in the landing folder."""
        if not self.incoming.is_dir():
            return []
        return sorted(
            path
            for path in self.incoming.iterdir()
            if path.is_file()
            and path.suffix.lower() in RAW_SUFFIXES
            and not path.name.startswith(".")
        )

    def clean(self) -> None:
        """Delete everything in the Live folder, including the input originals."""
        if not self.root.is_dir():
            return
        for child in self.root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
