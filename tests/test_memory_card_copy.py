"""Memory card copy workflow tests."""

from __future__ import annotations

import subprocess
from collections import namedtuple
from datetime import date
from pathlib import Path

import pytest

from photo_workflow import config as app_config
from photo_workflow import memory_card_copy


@pytest.fixture(autouse=True)
def mock_capture_date(monkeypatch, request) -> None:
    """Provide a capture date for byte-only media fixtures."""
    if request.node.name == "test_read_capture_date_uses_exiftool_datetime_original":
        return
    monkeypatch.setattr(
        memory_card_copy,
        "read_capture_date",
        lambda source_path: date(2026, 5, 29),
    )


def test_find_memory_card_mount_returns_volume_with_dcim(tmp_path: Path) -> None:
    """Verify card detection uses the presence of a DCIM directory."""
    mount_root = tmp_path / "Volumes"
    (mount_root / "Macintosh HD").mkdir(parents=True)
    card_root = mount_root / "UNTITLED"
    (card_root / "DCIM").mkdir(parents=True)

    detected_card = memory_card_copy.find_memory_card_mount(mount_root)

    assert detected_card == card_root


def test_run_memory_card_import_copies_flattens_and_cleans_card(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify ingest copies media files, flattens paths, and cleans the card."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    dcim_root = card_root / "DCIM" / "100MEDIA"
    dcim_root.mkdir(parents=True)
    (dcim_root / "A001.CR3").write_bytes(b"raw")
    (dcim_root / "A001.JPG").write_bytes(b"jpeg")
    camera_root = tmp_path / "camera"
    target_dir = camera_root / "2026" / "05" / "20260529"

    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3),
    )

    with caplog.at_level("INFO"):
        result = memory_card_copy.run_memory_card_import(
            camera_root,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
        )

    assert result is not None
    assert result.imported_files == 2
    assert result.target_dirs == (target_dir,)
    assert sorted(path.name for path in target_dir.iterdir()) == [
        "20260529_A001.CR3",
        "20260529_A001.JPG",
    ]
    assert list(memory_card_copy.iter_memory_card_files(card_root)) == []
    assert "detected memory card" in caplog.text
    assert "copied 2/2 files" in caplog.text
    assert "verifying 2 copied file(s) using basic verification" in caplog.text
    assert "ejected memory card" in caplog.text


def test_run_memory_card_import_ignores_ctg_files(tmp_path: Path, monkeypatch) -> None:
    """Verify default ignored Canon catalog files stay on the memory card."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    dcim_root = card_root / "DCIM" / "100MEDIA"
    dcim_root.mkdir(parents=True)
    (dcim_root / "A001.CR3").write_bytes(b"raw")
    ctg_path = dcim_root / "CANONMSC.CTG"
    ctg_path.write_bytes(b"catalog")
    camera_root = tmp_path / "camera"
    target_dir = camera_root / "2026" / "05" / "20260529"

    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3),
    )

    result = memory_card_copy.run_memory_card_import(
        camera_root,
        config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
    )

    assert result is not None
    assert result.imported_files == 1
    assert sorted(path.name for path in target_dir.iterdir()) == ["20260529_A001.CR3"]
    assert ctg_path.exists()


def test_run_memory_card_import_uses_configured_ignored_extensions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify configured ignored extensions are skipped during ingest."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    dcim_root = card_root / "DCIM" / "100MEDIA"
    dcim_root.mkdir(parents=True)
    (dcim_root / "A001.CR3").write_bytes(b"raw")
    ignored_path = dcim_root / "CARD.LOG"
    ignored_path.write_bytes(b"camera-log")
    camera_root = tmp_path / "camera"
    target_dir = camera_root / "2026" / "05" / "20260529"

    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3),
    )

    result = memory_card_copy.run_memory_card_import(
        camera_root,
        config=app_config.MemoryCardCopyConfig(
            card_mount_root=mount_root,
            ignored_extensions=(".log",),
        ),
    )

    assert result is not None
    assert result.imported_files == 1
    assert sorted(path.name for path in target_dir.iterdir()) == ["20260529_A001.CR3"]
    assert ignored_path.exists()


def test_run_memory_card_import_logs_copy_and_verification_elapsed_time(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify ingest logs elapsed time for copy and verification stages."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    dcim_root = card_root / "DCIM" / "100MEDIA"
    dcim_root.mkdir(parents=True)
    (dcim_root / "A001.CR3").write_bytes(b"raw")
    camera_root = tmp_path / "camera"
    timing_values = iter([10.0, 12.5, 20.0, 23.25, 30.0, 31.0, 40.0, 40.5])

    monkeypatch.setattr(memory_card_copy, "delete_memory_card_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    monkeypatch.setattr(memory_card_copy.time, "perf_counter", lambda: next(timing_values))

    with caplog.at_level("INFO"):
        memory_card_copy.run_memory_card_import(
            camera_root,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
            report_disk_space=False,
        )

    assert "import completed in 2.50s" in caplog.text
    assert "verification completed in 3.25s" in caplog.text


def test_build_copy_plan_sorts_files_by_capture_date(tmp_path: Path, monkeypatch) -> None:
    """Verify copied files use their capture date for directory and filename."""
    source_file = tmp_path / "AH9A9764.CR3"
    source_file.write_bytes(b"raw")
    camera_root = tmp_path / "camera"
    capture_date = date(2026, 6, 1)
    monkeypatch.setattr(
        memory_card_copy,
        "read_capture_date",
        lambda source_path: capture_date,
    )

    copy_plan = memory_card_copy.build_copy_plan([source_file], camera_root)

    assert copy_plan == [
        (
            source_file,
            camera_root / "2026" / "06" / "20260601" / "20260601_AH9A9764.CR3",
        ),
    ]


def test_build_copy_plan_logs_progress(tmp_path: Path, monkeypatch, caplog) -> None:
    """Verify copy-plan construction reports first, periodic, and final progress."""
    source_files = []
    for index in range(51):
        source_file = tmp_path / f"AH9A{index:04}.CR3"
        source_file.write_bytes(b"raw")
        source_files.append(source_file)
    monkeypatch.setattr(
        memory_card_copy,
        "read_capture_date",
        lambda source_path: date(2026, 6, 1),
    )

    with caplog.at_level("INFO"):
        memory_card_copy.build_copy_plan(source_files, tmp_path / "camera")

    assert "planning import destinations for 51 file(s)" in caplog.text
    assert "planned 1/51 files" in caplog.text
    assert "planned 50/51 files" in caplog.text
    assert "planned 51/51 files" in caplog.text


def test_read_capture_date_uses_exiftool_datetime_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify capture-date extraction reads DateTimeOriginal through exiftool."""
    source_file = tmp_path / "AH9A9764.CR3"
    source_file.write_bytes(b"raw")
    monkeypatch.setattr(memory_card_copy, "require_tool", lambda name: "/usr/local/bin/exiftool")
    monkeypatch.setattr(
        memory_card_copy.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "2026:06:01 14:30:00\n"),
    )

    capture_date = memory_card_copy.read_capture_date(source_file)

    assert capture_date == date(2026, 6, 1)


def test_build_copy_plan_rejects_existing_prefixed_target(tmp_path: Path) -> None:
    """Verify ingest fails when the prefixed target filename already exists."""
    source_file = tmp_path / "AH9A9764.CR3"
    source_file.write_bytes(b"raw")
    target_dir = tmp_path / "camera" / "2026" / "05" / "20260529"
    target_dir.mkdir(parents=True)
    (target_dir / "20260529_AH9A9764.CR3").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="Target file already exists"):
        memory_card_copy.build_copy_plan([source_file], tmp_path / "camera")


def test_run_memory_card_import_stops_before_delete_when_verification_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a failed verification leaves source files in place and skips eject."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    source_dir = card_root / "DCIM" / "100MEDIA"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "A001.CR3"
    source_file.write_bytes(b"raw")
    camera_root = tmp_path / "camera"
    eject_calls: list[Path] = []

    monkeypatch.setattr(
        memory_card_copy,
        "verify_copied_files",
        lambda copy_plan, verification_method: (_ for _ in ()).throw(ValueError("bad copy")),
    )
    monkeypatch.setattr(
        memory_card_copy,
        "eject_memory_card",
        lambda card_root: eject_calls.append(card_root),
    )

    with pytest.raises(ValueError, match="bad copy"):
        memory_card_copy.run_memory_card_import(
            camera_root,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
        )

    assert source_file.exists()
    assert eject_calls == []


def test_run_memory_card_import_stops_before_copy_when_target_lacks_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the import leaves the card unchanged when the target is too full."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    source_dir = card_root / "DCIM" / "100MEDIA"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "A001.CR3"
    source_file.write_bytes(b"raw")
    camera_root = tmp_path / "camera"
    usage = namedtuple("usage", ["total", "used", "free"])
    copied_files: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=100, used=98, free=2),
    )
    monkeypatch.setattr(
        memory_card_copy,
        "copy_files",
        lambda copy_plan: copied_files.extend(copy_plan),
    )

    with pytest.raises(OSError, match="insufficient free space"):
        memory_card_copy.run_memory_card_import(
            camera_root,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
        )

    assert copied_files == []
    assert source_file.exists()
    assert not camera_root.exists() or list(camera_root.iterdir()) == []


def test_run_memory_card_import_stops_before_copy_when_card_is_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a read-only card mount halts the import before any copy occurs."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    source_dir = card_root / "DCIM" / "100MEDIA"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "A001.CR3"
    source_file.write_bytes(b"raw")
    camera_root = tmp_path / "camera"
    copied_files: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        memory_card_copy.os,
        "statvfs",
        lambda _: type("FilesystemStats", (), {"f_flag": memory_card_copy.os.ST_RDONLY})(),
    )
    monkeypatch.setattr(
        memory_card_copy,
        "copy_files",
        lambda copy_plan: copied_files.extend(copy_plan),
    )

    with pytest.raises(OSError, match="memory card is read-only"):
        memory_card_copy.run_memory_card_import(
            camera_root,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
        )

    assert copied_files == []
    assert source_file.exists()
    assert not camera_root.exists()


def test_report_target_disk_space_warns_when_below_threshold(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify low free space emits a warning for the target volume."""
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=100 * 1024**3, used=96 * 1024**3, free=4 * 1024**3),
    )

    with caplog.at_level("INFO"):
        memory_card_copy.report_target_disk_space(
            tmp_path,
            config=app_config.MemoryCardCopyConfig(),
        )

    assert "target volume free space" in caplog.text
    assert "low disk space" in caplog.text


def test_report_target_disk_space_includes_purge_hint_when_space_is_reclaimable(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify low disk warnings include reclaimable-space guidance when available."""
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=100 * 1024**3, used=96 * 1024**3, free=4 * 1024**3),
    )

    with caplog.at_level("INFO"):
        memory_card_copy.report_target_disk_space(
            tmp_path,
            config=app_config.MemoryCardCopyConfig(),
            reclaimable_percent=8.52,
        )

    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]

    assert warning_messages == [
        f"low disk space on {tmp_path}: 4.0 GB free (4.0%)",
        "run with --purge-rejected to reclaim an additional 8.52% disk space.",
    ]


def test_format_eject_message_is_plain_when_not_interactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify non-interactive ejection messages omit ANSI styling."""
    monkeypatch.setattr(memory_card_copy.sys.stderr, "isatty", lambda: False)

    message = memory_card_copy.format_eject_message(tmp_path / "SDCARD")

    assert message == f"ejected memory card at {tmp_path / 'SDCARD'}"


def test_format_eject_message_is_green_when_interactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify interactive ejection messages use green ANSI styling."""
    monkeypatch.setattr(memory_card_copy.sys.stderr, "isatty", lambda: True)

    message = memory_card_copy.format_eject_message(tmp_path / "SDCARD")

    assert message == (
        f"{memory_card_copy.ANSI_SUCCESS}"
        f"ejected memory card at {tmp_path / 'SDCARD'}"
        f"{memory_card_copy.ANSI_RESET}"
    )


def test_verify_copied_files_logs_verification_start(tmp_path: Path, caplog) -> None:
    """Verify file verification logs the start message."""
    copy_plan: list[tuple[Path, Path]] = []
    for index in range(205):
        source_path = tmp_path / f"source-{index}.cr3"
        target_path = tmp_path / f"target-{index}.cr3"
        source_path.write_bytes(b"raw")
        target_path.write_bytes(b"raw")
        copy_plan.append((source_path, target_path))

    with caplog.at_level("INFO"):
        memory_card_copy.verify_copied_files(copy_plan, verification_method="basic")

    assert "verifying 205 copied file(s) using basic verification" in caplog.text
    assert "verified 1/205 files" not in caplog.text
    assert "verified 100/205 files" not in caplog.text
    assert "verified 200/205 files" not in caplog.text
    assert "verified 205/205 files" not in caplog.text
