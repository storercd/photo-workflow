"""Dashboard window and background worker tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from photo_workflow import config as app_config
from photo_workflow.dashboard import app as dashboard_app
from photo_workflow.rejected_folders import RejectedFolderAssessment, RejectedFolderAssessmentItem


def build_config(tmp_path: Path) -> app_config.AppConfig:
    """Return an AppConfig rooted under tmp_path for isolated dashboard tests."""
    return app_config.AppConfig(
        workflow=app_config.WorkflowConfig(camera_root=tmp_path / "camera"),
        memory_card_copy=app_config.MemoryCardCopyConfig(card_mount_root=tmp_path / "volumes"),
    )


def test_resolve_launcher_icon_falls_back_when_app_not_found(monkeypatch) -> None:
    """Verify the fallback theme icon is used when the app cannot be resolved."""
    monkeypatch.setattr(dashboard_app, "resolve_application_path", lambda name: None)

    icon = dashboard_app.resolve_launcher_icon(
        app_config.DashboardAppLauncher(name="Missing", app_name="Missing")
    )

    assert icon is not None


def test_launcher_button_click_launches_configured_app(monkeypatch) -> None:
    """Verify clicking a launcher button calls launch_app with its launcher."""
    app = QApplication.instance() or QApplication([])
    launched: list[app_config.DashboardAppLauncher] = []
    monkeypatch.setattr(dashboard_app, "launch_app", launched.append)
    monkeypatch.setattr(dashboard_app, "resolve_application_path", lambda name: None)

    launcher = app_config.DashboardAppLauncher(name="Aftershoot", app_name="Aftershoot")
    button = dashboard_app.LauncherButton(launcher)
    button.click()

    assert launched == [launcher]
    del app


def test_refresh_action_availability_enables_import_when_card_detected(tmp_path: Path) -> None:
    """Verify the import button enables only once a memory card is mounted."""
    app = QApplication.instance() or QApplication([])
    config = build_config(tmp_path)
    config.memory_card_copy.card_mount_root.mkdir(parents=True)
    window = dashboard_app.DashboardWindow(config)
    window.poll_timer.stop()

    assert window.import_button.isEnabled() is False

    card_root = config.memory_card_copy.card_mount_root / "NO_NAME"
    (card_root / "DCIM").mkdir(parents=True)
    window._refresh_action_availability()

    assert window.import_button.isEnabled() is True
    del app


def test_refresh_action_availability_enables_purge_when_rejected_folders_exist(
    tmp_path: Path,
) -> None:
    """Verify the purge button enables only once a _Rejected folder exists."""
    app = QApplication.instance() or QApplication([])
    config = build_config(tmp_path)
    config.memory_card_copy.card_mount_root.mkdir(parents=True)
    window = dashboard_app.DashboardWindow(config)
    window.poll_timer.stop()

    assert window.purge_button.isEnabled() is False

    rejected_dir = config.workflow.camera_root / "20260701" / "_Rejected"
    rejected_dir.mkdir(parents=True)
    (rejected_dir / "a.cr3").write_bytes(b"raw")
    window._refresh_action_availability()

    assert window.purge_button.isEnabled() is True
    del app


def test_purge_review_dialog_unchecking_folder_updates_selection(tmp_path: Path) -> None:
    """Verify unchecking a folder removes it from the selected paths and total."""
    app = QApplication.instance() or QApplication([])
    camera_root = tmp_path / "camera"
    first_path = camera_root / "20260701" / "_Rejected"
    second_path = camera_root / "20260702" / "_Rejected"
    assessment = RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=100,
        folders=[
            RejectedFolderAssessmentItem(
                folder_path=first_path, reclaimable_bytes=10, percent_of_disk=10.0
            ),
            RejectedFolderAssessmentItem(
                folder_path=second_path, reclaimable_bytes=5, percent_of_disk=5.0
            ),
        ],
        total_reclaimable_bytes=15,
        total_percent_of_disk=15.0,
    )

    dialog = dashboard_app.PurgeReviewDialog(assessment)
    assert dialog.selected_folder_paths() == {first_path, second_path}

    dialog.rows[0].checkbox.setChecked(False)

    assert dialog.selected_folder_paths() == {second_path}
    assert "Selected: 1 folder(s)" in dialog.total_label.text()
    del app


def test_import_worker_emits_stages_then_finished(monkeypatch, tmp_path: Path) -> None:
    """Verify ImportWorker reports each stage and then the final result."""
    config = build_config(tmp_path)
    expected_assessment = RejectedFolderAssessment(
        camera_root=tmp_path,
        disk_total_bytes=1,
        folders=[],
        total_reclaimable_bytes=0,
        total_percent_of_disk=0.0,
    )
    expected_dirs = (tmp_path / "camera" / "20260701",)

    def fake_run_full_import(config, *, on_stage=None):
        on_stage("Assessing rejected folders")
        on_stage("Importing memory card & generating video notes")
        on_stage("Reporting disk space")
        return expected_assessment, expected_dirs

    monkeypatch.setattr(dashboard_app, "run_full_import", fake_run_full_import)

    worker = dashboard_app.ImportWorker(config)
    stages: list[str] = []
    results: list[tuple] = []
    worker.signals.stage.connect(stages.append)
    worker.signals.finished.connect(lambda assessment, dirs: results.append((assessment, dirs)))

    worker.run()

    assert stages == list(dashboard_app.IMPORT_STAGES)
    assert results == [(expected_assessment, expected_dirs)]


def test_import_worker_emits_error_on_failure(monkeypatch, tmp_path: Path) -> None:
    """Verify ImportWorker reports failures via the error signal instead of raising."""
    config = build_config(tmp_path)

    def fake_run_full_import(config, *, on_stage=None):
        raise RuntimeError("card read failed")

    monkeypatch.setattr(dashboard_app, "run_full_import", fake_run_full_import)

    worker = dashboard_app.ImportWorker(config)
    errors: list[str] = []
    worker.signals.error.connect(errors.append)

    worker.run()

    assert errors == ["card read failed"]


def test_purge_worker_purges_only_selected_folders(tmp_path: Path) -> None:
    """Verify PurgeWorker deletes exactly the folders in the given assessment."""
    camera_root = tmp_path / "camera"
    keep_dir = camera_root / "20260701" / "_Rejected"
    purge_dir = camera_root / "20260702" / "_Rejected"
    keep_dir.mkdir(parents=True)
    purge_dir.mkdir(parents=True)

    assessment = RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=100,
        folders=[
            RejectedFolderAssessmentItem(
                folder_path=purge_dir, reclaimable_bytes=5, percent_of_disk=5.0
            ),
        ],
        total_reclaimable_bytes=5,
        total_percent_of_disk=5.0,
    )

    worker = dashboard_app.PurgeWorker(assessment)
    finished_counts: list[int] = []
    worker.signals.finished.connect(finished_counts.append)

    worker.run()

    assert finished_counts == [1]
    assert not purge_dir.exists()
    assert keep_dir.exists()
