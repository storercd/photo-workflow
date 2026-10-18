"""Read cached previews from a Lightroom Classic catalog."""

from __future__ import annotations

import sqlite3
from pathlib import Path

JPEG_MAGIC = b"\xff\xd8\xff"


def previews_dir_for(catalog_path: Path) -> Path:
    """Return the preview cache folder that sits beside a `.lrcat` catalog."""
    return catalog_path.with_name(f"{catalog_path.stem} Previews.lrdata")


def find_file_uuid(catalog_path: Path, raw_path: Path) -> str | None:
    """
    Look up the catalog file UUID that names a photo's cached previews.

    Returns:
        The `AgLibraryFile.id_global` value, or `None` if the photo is not cataloged.
    """
    query = """
        SELECT f.id_global, r.absolutePath, fo.pathFromRoot
        FROM AgLibraryFile f
        JOIN AgLibraryFolder fo ON fo.id_local = f.folder
        JOIN AgLibraryRootFolder r ON r.id_local = fo.rootFolder
        WHERE f.baseName = ? AND lower(f.extension) = ?
    """
    uri = f"{catalog_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=1.0)
    try:
        rows = connection.execute(
            query, (raw_path.stem, raw_path.suffix.lstrip(".").lower())
        ).fetchall()
    finally:
        connection.close()
    parent = raw_path.parent.resolve()
    for file_uuid, root, relative in rows:
        if (Path(root) / relative).resolve() == parent:
            return file_uuid
    return None


def best_cached_preview(
    previews_dir: Path,
    file_uuid: str,
    *,
    target_long_edge: int,
) -> Path | None:
    """
    Pick the most appropriate cached JPEG for one photo.

    Uses the most recently written render, then the smallest size that still
    reaches `target_long_edge`, otherwise the largest available.

    Returns:
        The preview file path, or `None` if Lightroom has not cached one.
    """
    folder = previews_dir / file_uuid[0] / file_uuid[:4]
    if not folder.is_dir():
        return None
    renders: dict[str, list[tuple[int, Path]]] = {}
    for path in folder.glob(f"{file_uuid}-*_*"):
        digest, _, size_text = path.name[len(file_uuid) + 1 :].rpartition("_")
        if size_text.isdigit():
            renders.setdefault(digest, []).append((int(size_text), path))
    if not renders:
        return None
    sizes = max(
        renders.values(),
        key=lambda entries: max(path.stat().st_mtime for _, path in entries),
    )
    sizes.sort()
    for size, path in sizes:
        if size >= target_long_edge:
            return path
    return sizes[-1][1]


def find_lightroom_preview(
    catalog_path: Path,
    raw_path: Path,
    *,
    target_long_edge: int,
) -> Path | None:
    """
    Return the cached preview file Lightroom holds for a RAW photo, if any.

    Returns:
        The preview file path, or `None` if the photo or its preview is missing.
    """
    file_uuid = find_file_uuid(catalog_path, raw_path)
    if file_uuid is None:
        return None
    return best_cached_preview(
        previews_dir_for(catalog_path), file_uuid, target_long_edge=target_long_edge
    )
