"""JPEG companion-file purge tests."""

from __future__ import annotations

from pathlib import Path

from photo_workflow import jpeg_purge


def test_assess_jpegs_finds_jpegs_recursively_and_totals_their_size(tmp_path: Path) -> None:
    """Verify the assessment includes case-insensitive JPEG files only."""
    root_dir = tmp_path / "camera"
    nested_dir = root_dir / "20260918"
    nested_dir.mkdir(parents=True)
    first_jpeg = root_dir / "IMG_0001.jpg"
    second_jpeg = nested_dir / "IMG_0002.JPG"
    first_jpeg.write_bytes(b"1234")
    second_jpeg.write_bytes(b"123456")
    (nested_dir / "IMG_0002.cr3").write_bytes(b"raw")

    assessment = jpeg_purge.assess_jpegs(root_dir)

    assert assessment.root_dir == root_dir
    assert assessment.jpeg_paths == [second_jpeg, first_jpeg]
    assert assessment.reclaimable_bytes == 10


def test_purge_jpegs_dry_run_leaves_files_in_place(tmp_path: Path) -> None:
    """Verify dry-run reports deletions without changing the filesystem."""
    jpeg_path = tmp_path / "camera" / "IMG_0001.jpg"
    jpeg_path.parent.mkdir(parents=True)
    jpeg_path.write_bytes(b"1234")

    result = jpeg_purge.purge_jpegs(
        jpeg_purge.assess_jpegs(jpeg_path.parent),
        dry_run=True,
    )

    assert result == jpeg_purge.JpegPurgeResult(deleted_files=0, reclaimed_bytes=0)
    assert jpeg_path.exists()


def test_purge_jpegs_deletes_assessed_files(tmp_path: Path) -> None:
    """Verify purge deletes each assessed JPEG file and reports reclaimed bytes."""
    root_dir = tmp_path / "camera"
    jpeg_path = root_dir / "20260918" / "IMG_0001.jpg"
    jpeg_path.parent.mkdir(parents=True)
    jpeg_path.write_bytes(b"1234")

    result = jpeg_purge.purge_jpegs(jpeg_purge.assess_jpegs(root_dir), dry_run=False)

    assert result.deleted_files == 1
    assert result.reclaimed_bytes == 4
    assert not jpeg_path.exists()
