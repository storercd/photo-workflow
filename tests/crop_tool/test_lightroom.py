"""Lightroom catalog lookup tests."""

import sqlite3
from pathlib import Path

from photo_workflow.crop_tool.lightroom import xmp_is_dirty, xmp_lacks_catalog_edits


def _catalog(path: Path, root: Path, dirty: int) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE AgLibraryFile (id_local INTEGER, baseName, extension, folder);
        CREATE TABLE Adobe_images (id_local INTEGER, rootFile INTEGER);
        CREATE TABLE AgLibraryFolder (id_local INTEGER, rootFolder, pathFromRoot);
        CREATE TABLE AgLibraryRootFolder (id_local INTEGER, absolutePath);
        CREATE TABLE Adobe_AdditionalMetadata (image INTEGER, externalXmpIsDirty INTEGER);
        """
    )
    connection.execute("INSERT INTO AgLibraryFile VALUES (1, 'IMG_1', 'cr3', 1)")
    connection.execute("INSERT INTO Adobe_images VALUES (7, 1)")
    connection.execute("INSERT INTO AgLibraryFolder VALUES (1, 1, 'shoot/')")
    connection.execute("INSERT INTO AgLibraryRootFolder VALUES (1, ?)", (f"{root}/",))
    connection.execute("INSERT INTO Adobe_AdditionalMetadata VALUES (7, ?)", (dirty,))
    connection.commit()
    connection.close()


def test_xmp_is_dirty_reads_lightroom_flag(tmp_path: Path) -> None:
    """The catalog's externalXmpIsDirty flag maps to the sidecar being out of date."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    catalog = tmp_path / "test.lrcat"
    _catalog(catalog, tmp_path, 1)

    assert xmp_is_dirty(catalog, folder / "IMG_1.CR3") is True


def test_xmp_is_dirty_false_when_synced(tmp_path: Path) -> None:
    """A clean flag reports the sidecar as in sync."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    catalog = tmp_path / "test.lrcat"
    _catalog(catalog, tmp_path, 0)

    assert xmp_is_dirty(catalog, folder / "IMG_1.CR3") is False


def test_xmp_is_dirty_none_when_uncataloged(tmp_path: Path) -> None:
    """Photos missing from the catalog report no status."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    catalog = tmp_path / "test.lrcat"
    _catalog(catalog, tmp_path, 1)

    assert xmp_is_dirty(catalog, folder / "OTHER.CR3") is None


def _develop_catalog(path: Path, root: Path, text: str) -> None:
    _catalog(path, root, 0)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE Adobe_imageDevelopSettings (image INTEGER, text)")
    connection.execute("INSERT INTO Adobe_imageDevelopSettings VALUES (7, ?)", (text,))
    connection.commit()
    connection.close()


def test_edits_missing_from_xmp_without_develop_settings(tmp_path: Path) -> None:
    """A sidecar with no develop settings is stale when the catalog has edits."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    catalog = tmp_path / "test.lrcat"
    _develop_catalog(catalog, tmp_path, "s = {\nExposure2012 = 0.33,\nWhites2012 = -16 }")
    xmp = folder / "IMG_1.xmp"
    xmp.write_text("<x:xmpmeta xmlns:x='adobe:ns:meta/'/>")

    assert xmp_lacks_catalog_edits(catalog, folder / "IMG_1.CR3", xmp) is True


def test_unedited_catalog_matches_empty_xmp(tmp_path: Path) -> None:
    """Default-valued catalog settings do not flag a bare sidecar."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    catalog = tmp_path / "test.lrcat"
    _develop_catalog(catalog, tmp_path, "s = {\nExposure2012 = 0 }")

    assert xmp_lacks_catalog_edits(catalog, folder / "IMG_1.CR3", folder / "IMG_1.xmp") is False
