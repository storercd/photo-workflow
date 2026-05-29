"""Memory card copy workflow tests."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from photo_workflow import config as app_config
from photo_workflow import memory_card_copy


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
    target_dir = tmp_path / "camera" / "20260529"

    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3),
    )

    with caplog.at_level("INFO"):
        result = memory_card_copy.run_memory_card_import(
            target_dir,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
        )

    assert result is not None
    assert result.imported_files == 2
    assert sorted(path.name for path in target_dir.iterdir()) == ["A001.CR3", "A001.JPG"]
    assert list(memory_card_copy.iter_memory_card_files(card_root)) == []
    assert "detected memory card" in caplog.text
    assert "copied 2/2 files" in caplog.text
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
    target_dir = tmp_path / "camera" / "20260529"

    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3),
    )

    result = memory_card_copy.run_memory_card_import(
        target_dir,
        config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
    )

    assert result is not None
    assert result.imported_files == 1
    assert sorted(path.name for path in target_dir.iterdir()) == ["A001.CR3"]
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
    target_dir = tmp_path / "camera" / "20260529"

    monkeypatch.setattr(memory_card_copy, "eject_memory_card", lambda card_root: None)
    usage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        memory_card_copy.shutil,
        "disk_usage",
        lambda _: usage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3),
    )

    result = memory_card_copy.run_memory_card_import(
        target_dir,
        config=app_config.MemoryCardCopyConfig(
            card_mount_root=mount_root,
            ignored_extensions=(".log",),
        ),
    )

    assert result is not None
    assert result.imported_files == 1
    assert sorted(path.name for path in target_dir.iterdir()) == ["A001.CR3"]
    assert ignored_path.exists()


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
    target_dir = tmp_path / "camera" / "20260529"
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
            target_dir,
            config=app_config.MemoryCardCopyConfig(card_mount_root=mount_root),
        )

    assert source_file.exists()
    assert eject_calls == []


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
