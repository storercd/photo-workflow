"""Workflow orchestration tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from photo_workflow import config as app_config
from photo_workflow import rejected_folders, workflow


def build_assessment(camera_root: Path) -> rejected_folders.RejectedFolderAssessment:
    """Return a minimal rejected-folder assessment for workflow tests."""
    return rejected_folders.RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=100,
        folders=[],
        total_reclaimable_bytes=15,
        total_percent_of_disk=15.0,
    )


def build_non_empty_assessment(camera_root: Path) -> rejected_folders.RejectedFolderAssessment:
    """Return a rejected-folder assessment with one folder for purge-path tests."""
    return rejected_folders.RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=100,
        folders=[
            rejected_folders.RejectedFolderAssessmentItem(
                folder_path=camera_root / "20260701" / "_Rejected",
                reclaimable_bytes=15,
                percent_of_disk=15.0,
            )
        ],
        total_reclaimable_bytes=15,
        total_percent_of_disk=15.0,
    )


def test_open_target_folder_launches_finder_for_processed_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the processed folder opens in Finder after the workflow completes."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    workflow.open_target_folder(tmp_path / "camera" / "20260529")

    assert commands == [["open", str(tmp_path / "camera" / "20260529")]]


def test_main_runs_memory_card_import_before_video_notes(tmp_path: Path, monkeypatch) -> None:
    """Verify the full workflow runs card import before video note generation."""
    source_dir = tmp_path / "camera" / "20260529"
    call_order: list[str] = []

    monkeypatch.setattr(
        workflow,
        "load_config",
        lambda: app_config.AppConfig(
            workflow=app_config.WorkflowConfig(camera_root=tmp_path / "camera"),
            memory_card_copy=app_config.MemoryCardCopyConfig(),
            video_notes=app_config.VideoNotesConfig(max_duration_seconds=9.0),
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_today_source_dir",
        lambda today=None, camera_root=None: source_dir,
    )
    monkeypatch.setattr(
        workflow,
        "run_memory_card_import",
        lambda target_dir, config, report_disk_space: call_order.append("import"),
    )
    monkeypatch.setattr(
        workflow,
        "run_video_notes_step",
        lambda source_dir, config: call_order.append("video_notes"),
    )
    monkeypatch.setattr(
        workflow,
        "run_rejected_folder_step",
        lambda camera_root, purge_rejected: call_order.append("rejected")
        or build_assessment(camera_root),
    )
    monkeypatch.setattr(
        workflow,
        "report_target_disk_space",
        lambda target_dir, config, reclaimable_percent=None: call_order.append("disk_space"),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_disk_usage_path",
        lambda camera_root: camera_root,
    )
    monkeypatch.setattr(
        workflow,
        "open_target_folder",
        lambda target_dir: call_order.append("open"),
    )

    workflow.main([])

    assert call_order == ["import", "video_notes", "rejected", "disk_space", "open"]


def test_main_passes_purge_rejected_flag_to_rejected_folder_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the CLI forwards --purge-rejected to the cleanup step."""
    source_dir = tmp_path / "camera" / "20260529"
    purge_values: list[bool] = []

    monkeypatch.setattr(
        workflow,
        "load_config",
        lambda: app_config.AppConfig(
            workflow=app_config.WorkflowConfig(camera_root=tmp_path / "camera"),
            memory_card_copy=app_config.MemoryCardCopyConfig(),
            video_notes=app_config.VideoNotesConfig(max_duration_seconds=9.0),
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_today_source_dir",
        lambda today=None, camera_root=None: source_dir,
    )
    monkeypatch.setattr(
        workflow,
        "run_memory_card_import",
        lambda target_dir, config, report_disk_space: None,
    )
    monkeypatch.setattr(workflow, "run_video_notes_step", lambda source_dir, config: None)
    monkeypatch.setattr(
        workflow,
        "run_rejected_folder_step",
        lambda camera_root, purge_rejected: purge_values.append(purge_rejected)
        or build_assessment(camera_root),
    )
    monkeypatch.setattr(
        workflow,
        "report_target_disk_space",
        lambda target_dir, config, reclaimable_percent=None: None,
    )
    monkeypatch.setattr(workflow, "resolve_disk_usage_path", lambda camera_root: camera_root)
    monkeypatch.setattr(workflow, "open_target_folder", lambda target_dir: None)

    workflow.main(["--purge-rejected"])

    assert purge_values == [True]


def test_main_defers_disk_space_report_until_after_rejected_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the workflow reports disk space after the rejected-folder step."""
    source_dir = tmp_path / "camera" / "20260529"
    call_order: list[str] = []

    monkeypatch.setattr(
        workflow,
        "load_config",
        lambda: app_config.AppConfig(
            workflow=app_config.WorkflowConfig(camera_root=tmp_path / "camera"),
            memory_card_copy=app_config.MemoryCardCopyConfig(),
            video_notes=app_config.VideoNotesConfig(max_duration_seconds=9.0),
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_today_source_dir",
        lambda today=None, camera_root=None: source_dir,
    )
    monkeypatch.setattr(
        workflow,
        "run_memory_card_import",
        lambda target_dir, config, report_disk_space: call_order.append(
            f"import:{report_disk_space}"
        ),
    )
    monkeypatch.setattr(
        workflow,
        "run_video_notes_step",
        lambda source_dir, config: call_order.append("video_notes"),
    )
    monkeypatch.setattr(
        workflow,
        "run_rejected_folder_step",
        lambda camera_root, purge_rejected: call_order.append("rejected")
        or build_assessment(camera_root),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_disk_usage_path",
        lambda camera_root: call_order.append("resolve_disk") or camera_root,
    )
    monkeypatch.setattr(
        workflow,
        "report_target_disk_space",
        lambda target_dir, config, reclaimable_percent=None: call_order.append("disk_space"),
    )
    monkeypatch.setattr(workflow, "open_target_folder", lambda target_dir: None)

    workflow.main([])

    assert call_order == [
        "import:False",
        "video_notes",
        "rejected",
        "resolve_disk",
        "disk_space",
    ]


def test_main_passes_reclaimable_percent_to_final_disk_space_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the final disk-space report receives reclaimable rejected-folder space."""
    source_dir = tmp_path / "camera" / "20260529"
    captured_percentages: list[float | None] = []

    monkeypatch.setattr(
        workflow,
        "load_config",
        lambda: app_config.AppConfig(
            workflow=app_config.WorkflowConfig(camera_root=tmp_path / "camera"),
            memory_card_copy=app_config.MemoryCardCopyConfig(),
            video_notes=app_config.VideoNotesConfig(max_duration_seconds=9.0),
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_today_source_dir",
        lambda today=None, camera_root=None: source_dir,
    )
    monkeypatch.setattr(
        workflow,
        "run_memory_card_import",
        lambda target_dir, config, report_disk_space: None,
    )
    monkeypatch.setattr(workflow, "run_video_notes_step", lambda source_dir, config: None)
    monkeypatch.setattr(
        workflow,
        "run_rejected_folder_step",
        lambda camera_root, purge_rejected: build_assessment(camera_root),
    )
    monkeypatch.setattr(workflow, "resolve_disk_usage_path", lambda camera_root: camera_root)
    monkeypatch.setattr(
        workflow,
        "report_target_disk_space",
        lambda target_dir, config, reclaimable_percent=None: captured_percentages.append(
            reclaimable_percent
        ),
    )
    monkeypatch.setattr(workflow, "open_target_folder", lambda target_dir: None)

    workflow.main([])

    assert captured_percentages == [15.0]


def test_main_logs_video_notes_elapsed_time(tmp_path: Path, monkeypatch, caplog) -> None:
    """Verify the workflow logs elapsed time for the video-notes stage."""
    source_dir = tmp_path / "camera" / "20260529"
    timing_values = iter([50.0, 54.75])

    monkeypatch.setattr(
        workflow,
        "load_config",
        lambda: app_config.AppConfig(
            workflow=app_config.WorkflowConfig(camera_root=tmp_path / "camera"),
            memory_card_copy=app_config.MemoryCardCopyConfig(),
            video_notes=app_config.VideoNotesConfig(max_duration_seconds=9.0),
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_today_source_dir",
        lambda today=None, camera_root=None: source_dir,
    )
    monkeypatch.setattr(
        workflow,
        "run_memory_card_import",
        lambda target_dir, config, report_disk_space: None,
    )
    monkeypatch.setattr(workflow, "run_video_notes_step", lambda source_dir, config: None)
    monkeypatch.setattr(
        workflow,
        "run_rejected_folder_step",
        lambda camera_root, purge_rejected: build_assessment(camera_root),
    )
    monkeypatch.setattr(workflow, "resolve_disk_usage_path", lambda camera_root: camera_root)
    monkeypatch.setattr(
        workflow,
        "report_target_disk_space",
        lambda target_dir, config, reclaimable_percent=None: None,
    )
    monkeypatch.setattr(workflow, "open_target_folder", lambda target_dir: None)
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(timing_values))

    with caplog.at_level("INFO"):
        workflow.main([])

    assert "video notes completed in 4.75s" in caplog.text


def test_run_rejected_folder_step_logs_scan_and_purge_elapsed_time(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """Verify rejected-folder scan and purge stages log elapsed time."""
    timing_values = iter([1.0, 2.25, 5.0, 6.5])

    monkeypatch.setattr(
        workflow,
        "assess_rejected_folders",
        lambda camera_root: build_non_empty_assessment(camera_root),
    )
    monkeypatch.setattr(
        workflow,
        "log_rejected_folder_assessment",
        lambda assessment, purge_rejected: None,
    )
    monkeypatch.setattr(workflow, "purge_rejected_folders", lambda assessment: 1)
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(timing_values))

    with caplog.at_level("INFO"):
        workflow.run_rejected_folder_step(tmp_path / "camera", purge_rejected=True)

    assert "rejected-folder scan completed in 1.25s" in caplog.text
    assert "rejected-folder purge completed in 1.50s" in caplog.text
