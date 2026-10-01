"""Rejected-folder assessment and purge tests."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

from photo_workflow import rejected_folders


def test_assess_rejected_folders_reports_each_folder_and_total(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify assessment reports per-folder and total reclaimable disk usage."""
    camera_root = tmp_path / "camera"
    first_rejected = camera_root / "20260701" / "_Rejected"
    second_rejected = camera_root / "20260702" / "_Rejected"
    first_rejected.mkdir(parents=True)
    second_rejected.mkdir(parents=True)
    (first_rejected / "a.cr3").write_bytes(b"1234567890")
    (second_rejected / "b.cr3").write_bytes(b"12345")

    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        rejected_folders.shutil,
        "disk_usage",
        lambda _: usage(total=100, used=40, free=60),
    )

    assessment = rejected_folders.assess_rejected_folders(camera_root)

    assert [item.folder_path for item in assessment.folders] == [
        first_rejected,
        second_rejected,
    ]
    assert [item.reclaimable_bytes for item in assessment.folders] == [10, 5]
    assert [item.percent_of_disk for item in assessment.folders] == [10.0, 5.0]
    assert assessment.total_reclaimable_bytes == 15
    assert assessment.total_percent_of_disk == 15.0


def test_purge_rejected_folders_deletes_all_assessed_folders(tmp_path: Path) -> None:
    """Verify purge removes each assessed rejected folder completely."""
    camera_root = tmp_path / "camera"
    rejected_dir = camera_root / "20260701" / "_Rejected"
    nested_dir = rejected_dir / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "a.cr3").write_bytes(b"raw")

    assessment = rejected_folders.assess_rejected_folders(camera_root)

    purged_count = rejected_folders.purge_rejected_folders(assessment)

    assert purged_count == 1
    assert not rejected_dir.exists()


def test_log_rejected_folder_assessment_reports_summary(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify assessment logging shows each folder and the total reclaimable space."""
    camera_root = tmp_path / "camera"
    rejected_dir = camera_root / "20260701" / "_Rejected"
    rejected_dir.mkdir(parents=True)
    (rejected_dir / "a.cr3").write_bytes(b"1234567890")

    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        rejected_folders.shutil,
        "disk_usage",
        lambda _: usage(total=10 * 1024**3, used=5 * 1024**3, free=5 * 1024**3),
    )

    assessment = rejected_folders.assess_rejected_folders(camera_root)

    with caplog.at_level("INFO"):
        rejected_folders.log_rejected_folder_assessment(assessment, purge_rejected=False)

    assert "rejected folder assessment under" in caplog.text
    assert "20260701/_Rejected" in caplog.text
    assert "total rejected folders: 1" in caplog.text


def test_log_rejected_folder_assessment_reports_when_none_found(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify assessment logging handles the absence of rejected folders."""
    camera_root = tmp_path / "camera"
    camera_root.mkdir(parents=True)

    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        rejected_folders.shutil,
        "disk_usage",
        lambda _: usage(total=10 * 1024**3, used=5 * 1024**3, free=5 * 1024**3),
    )

    assessment = rejected_folders.assess_rejected_folders(camera_root)

    with caplog.at_level("INFO"):
        rejected_folders.log_rejected_folder_assessment(assessment, purge_rejected=True)

    assert "no _Rejected folders found" in caplog.text


def test_log_rejected_folder_assessment_reports_green_purge_summary(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify purge mode logs a green summary line for reclaimable disk space."""
    camera_root = tmp_path / "camera"
    rejected_dir = camera_root / "20260701" / "_Rejected"
    rejected_dir.mkdir(parents=True)
    (rejected_dir / "a.cr3").write_bytes(b"1234567890")

    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        rejected_folders.shutil,
        "disk_usage",
        lambda _: usage(total=100, used=40, free=60),
    )
    monkeypatch.setattr(rejected_folders.sys.stderr, "isatty", lambda: True)

    assessment = rejected_folders.assess_rejected_folders(camera_root)

    with caplog.at_level("INFO"):
        rejected_folders.log_rejected_folder_assessment(assessment, purge_rejected=True)

    assert "purging 10.00% disk space from rejected folders" in caplog.text


def test_select_assessment_folders_filters_to_chosen_paths(tmp_path: Path) -> None:
    """Verify selecting a subset recomputes totals for only the chosen folders."""
    camera_root = tmp_path / "camera"
    first_item = rejected_folders.RejectedFolderAssessmentItem(
        folder_path=camera_root / "20260701" / "_Rejected",
        reclaimable_bytes=10,
        percent_of_disk=10.0,
    )
    second_item = rejected_folders.RejectedFolderAssessmentItem(
        folder_path=camera_root / "20260702" / "_Rejected",
        reclaimable_bytes=5,
        percent_of_disk=5.0,
    )
    assessment = rejected_folders.RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=100,
        folders=[first_item, second_item],
        total_reclaimable_bytes=15,
        total_percent_of_disk=15.0,
    )

    selected = rejected_folders.select_assessment_folders(assessment, {first_item.folder_path})

    assert selected.folders == [first_item]
    assert selected.total_reclaimable_bytes == 10
    assert selected.total_percent_of_disk == 10.0


def test_select_assessment_folders_returns_empty_for_no_selection(tmp_path: Path) -> None:
    """Verify an empty selection yields zeroed totals and no folders."""
    camera_root = tmp_path / "camera"
    item = rejected_folders.RejectedFolderAssessmentItem(
        folder_path=camera_root / "20260701" / "_Rejected",
        reclaimable_bytes=10,
        percent_of_disk=10.0,
    )
    assessment = rejected_folders.RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=100,
        folders=[item],
        total_reclaimable_bytes=10,
        total_percent_of_disk=10.0,
    )

    selected = rejected_folders.select_assessment_folders(assessment, set())

    assert selected.folders == []
    assert selected.total_reclaimable_bytes == 0
    assert selected.total_percent_of_disk == 0.0


def test_format_purge_message_is_green_when_interactive(monkeypatch) -> None:
    """Verify interactive purge summaries use green ANSI styling."""
    monkeypatch.setattr(rejected_folders.sys.stderr, "isatty", lambda: True)

    message = rejected_folders.format_purge_message(8.52)

    assert message == (
        f"{rejected_folders.ANSI_SUCCESS}"
        "purging 8.52% disk space from rejected folders"
        f"{rejected_folders.ANSI_RESET}"
    )
