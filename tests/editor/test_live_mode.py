"""Tests for Live mode in the editor window."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from photo_workflow.editor.app import CropWindow
from photo_workflow.editor.live_ui import IdleMonitor
from photo_workflow.editor.metadata import PhotoMetadata
from photo_workflow.editor.model import CropRect
from photo_workflow.live import settings as live_settings
from photo_workflow.live.features.mask import NO_SUBJECT_MESSAGE
from photo_workflow.live.stages import StageId, StageStatus


@pytest.fixture(autouse=True)
def no_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these tests away from the real system keychain."""
    monkeypatch.setattr(live_settings.KeychainSecretStore, "get", lambda self, name: "")
    monkeypatch.setattr(live_settings.KeychainSecretStore, "set", lambda self, name, value: None)


def make_live_photo(live_root: Path) -> Path:
    """
    Create a renamed fake original in a Live date folder.

    Returns:
        The photo path.
    """
    photo = live_root / "2025" / "10" / "20251031" / "input" / "20251031_101500_00_IMG_1.CR3"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"raw")
    return photo


def make_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CropWindow:
    """
    Create a window whose Live folder is `tmp_path/live`, without decoding previews.

    Returns:
        The window.
    """
    QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = CropWindow(settings=settings, camera_root=tmp_path / "camera")
    monkeypatch.setattr(window, "_start_photo_load", lambda path: None)
    return window


def test_live_mode_lists_live_photos_and_keeps_crop_review_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode opens photos from the Live folder, and Crop/Review still toggles."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)

    window._toggle_live_mode()

    assert "Live Crop Mode" in window.windowTitle()
    assert window._photos == [photo]
    assert not window.live_dock.isHidden()
    window._toggle_review_mode()
    assert "Live Review Mode" in window.windowTitle()
    window._toggle_live_mode()
    assert window.live_dock.isHidden()
    assert "Live" not in window.windowTitle()
    window.close()


def test_status_panel_toggle_persists_and_tab_hides_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Feature toggles are stored; the edge tab hides and shows the panel."""
    make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    panel = window.live_dock.panel

    panel.toggles[StageId.DENOISE].setChecked(True)

    assert window._live.pipeline.settings.is_enabled(StageId.DENOISE)
    assert window._settings.value("live/enabled", "", type=str) == "denoise"
    assert panel.toggles[StageId.AUTOCROP].isEnabled()
    assert panel.toggles[StageId.AUTOCROP].isChecked()
    window.live_dock.toggle_panel()
    assert panel.isHidden()
    window.live_dock.toggle_panel()
    assert not panel.isHidden()
    window.close()


def test_failed_photo_is_grayed_with_warning_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A photo whose stage failed shows the large failure overlay."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    pipeline = window._live.pipeline

    record = pipeline._state.images[photo][StageId.DENOISE]
    record.status, record.error = StageStatus.FAILED, "boom"
    window._refresh_live_failure()

    assert window.view._failed is not None
    assert "boom" in window.view._failed
    window.close()


def test_clean_wipes_live_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Clean action empties the Live folder, input included."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()

    window._live.pipeline.clean()
    window._live_cleaned("")

    assert not photo.exists()
    assert window._photos == []
    window.close()


def test_status_panel_shows_stage_info_only_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running service's status line (the FTP address) shows under its stage row."""
    make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    panel = window.live_dock.panel
    snapshot = window._live.pipeline.snapshot()

    panel.update_status(snapshot, [], blink_on=False, info={StageId.FTP: "FTP at 10.0.0.2:2121"})
    assert panel.info_labels[StageId.FTP].text() == "FTP at 10.0.0.2:2121"
    assert not panel.info_labels[StageId.FTP].isHidden()

    panel.update_status(snapshot, [], blink_on=False)
    assert panel.info_labels[StageId.FTP].isHidden()
    window.close()


def test_no_subject_badge_shows_only_for_a_mask_skipped_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The badge needs Mask on and Mask skipped for 'no subject' on the current photo."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    pipeline = window._live.pipeline
    window._current_path = photo
    skipped = {"message": NO_SUBJECT_MESSAGE}
    monkeypatch.setattr(pipeline, "skip_message", lambda *_: skipped["message"])
    monkeypatch.setattr(pipeline, "progress", lambda _photo: "")

    window._refresh_live_failure()
    assert window.view._badge is None  # Mask is off

    pipeline.set_settings(pipeline.settings.with_enabled(StageId.MASK, True))
    window._refresh_live_failure()
    assert window.view._badge is not None
    assert "no subject" in window.view._badge

    skipped["message"] = ""
    window._refresh_live_failure()
    assert window.view._badge is None
    window.close()


def test_idle_message_opens_and_creates_the_live_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty Live message is a link that creates and opens the watched folder."""
    opened: list[str] = []
    monkeypatch.setattr(
        "photo_workflow.editor.app.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )
    window = make_window(tmp_path, monkeypatch)
    label = window.position_count_label

    window._open_live_folder()
    assert opened == []

    window._toggle_live_mode()
    root = window._live.pipeline.layout.root
    assert label.text() == "Waiting for Live photos"
    assert label.cursor().shape() == Qt.CursorShape.PointingHandCursor

    label.clicked.emit()

    assert root.is_dir()
    assert opened == [str(root)]

    make_live_photo(root)
    window._live.pipeline.scan()
    window.open_folder(root)
    assert label.cursor().shape() == Qt.CursorShape.PointingHandCursor
    window._open_live_folder()
    assert opened == [str(root), str(root)]
    window.close()


def test_crop_source_indicator_follows_auto_crop_until_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An auto crop is marked in Live mode and becomes manual once the user edits it."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    window._current_path = photo
    metadata = PhotoMetadata(6000, 4000, CropRect(0.1, 0.1, 0.9, 0.9), auto_crop=True)
    window._current_metadata = metadata
    window._metadata_cache[photo] = metadata

    window._refresh_crop_source()
    assert window.view.crop_source == "auto"

    window._mark_dirty()
    assert window.view.crop_source == "manual"
    assert window._metadata_cache[photo].auto_crop is False

    window._current_metadata = PhotoMetadata(6000, 4000, CropRect(0, 0, 1, 1))
    window._refresh_crop_source()
    assert window.view.crop_source == "manual"  # unmarked crops count as manual

    window._live_mode = False
    window._refresh_crop_source()
    assert window.view.crop_source is None
    window.close()


def test_clicking_the_stage_error_flag_shows_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The red ! beside a halted feature opens a window with the full error text."""
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    panel = window.live_dock.panel
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append(text))
    window._live.pipeline.halt(StageId.FTP, "FTP username is not configured")
    window._update_live_panel()

    panel.error_labels[StageId.FTP].click()

    assert shown
    assert "FTP username is not configured" in shown[0]
    window.close()


def test_idle_monitor_holds_while_active_and_releases_after_a_quiet_spell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the mouse or pressing a key holds the pipeline; stillness releases it."""
    make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    pipeline = window._live.pipeline
    now = [100.0]
    cursor = [QPoint(0, 0)]
    monitor = IdleMonitor(pipeline, clock=lambda: now[0], cursor=lambda: cursor[0])

    monitor.configure(replace(pipeline.settings, idle_processing=True, idle_delay_seconds=3.0))
    assert pipeline.held

    now[0] += 3.5
    monitor.check()
    assert not pipeline.held

    cursor[0] = QPoint(5, 5)
    monitor.check()
    assert pipeline.held

    now[0] += 3.5
    monitor.check()
    assert not pipeline.held
    monitor.eventFilter(
        window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
    )
    assert pipeline.held

    panel = window.live_dock.panel
    panel.update_status(pipeline.snapshot(), [], blink_on=False)
    assert not panel.held_label.isHidden()

    monitor.configure(replace(pipeline.settings, idle_processing=False))
    assert not pipeline.held
    window.close()


def test_idle_button_toggles_and_persists_idle_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status panel button switches Idle Processing, and settings changes update it."""
    make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    panel = window.live_dock.panel
    pipeline = window._live.pipeline
    assert panel.idle_button.text() == "Idle Processing"
    assert not panel.idle_button.isChecked()

    panel.idle_button.click()
    assert window._live_settings.idle_processing
    assert pipeline.held
    assert live_settings.load_live_settings(window._settings).idle_processing

    window._apply_live_settings(replace(window._live_settings, idle_processing=False))
    assert not panel.idle_button.isChecked()
    assert not pipeline.held
    window.close()


def test_viewed_photo_is_prioritized_in_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting a photo in Live mode makes the pipeline serve it first."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()

    window._show_photo(0)

    assert window._live.pipeline._priority == window._photos[0]
    assert photo.name == window._photos[0].name
    window.close()


def test_saving_a_crop_reruns_export_and_cloud_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manual crop edit makes the exported JPEG stale, so Export and Cloud Sync redo it."""
    photo = make_live_photo(tmp_path / "live")
    window = make_window(tmp_path, monkeypatch)
    window._toggle_live_mode()
    pipeline = window._live.pipeline
    for stage in StageId:
        pipeline._state.images[photo][stage].status = StageStatus.DONE

    window._save_finished(str(photo), 1, CropRect(0.1, 0.1, 0.9, 0.9), 0.0, "")

    assert pipeline.status(photo, StageId.AUTOCROP) is StageStatus.DONE
    assert pipeline.status(photo, StageId.EXPORT) is StageStatus.PENDING
    assert pipeline.status(photo, StageId.CLOUD_SYNC) is StageStatus.PENDING
    window.close()


def test_white_balance_override_reaches_the_live_lighting_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixing the color temperature in the editor changes the Live setting, then clears it."""
    window = make_window(tmp_path, monkeypatch)
    window._open_live()
    window._wb_override = 4300.0

    window._apply_wb_override()
    assert window._live is not None
    assert window._live.pipeline.settings.color_temperature == 4300.0

    window._wb_override = None
    window._apply_wb_override()
    assert window._live.pipeline.settings.color_temperature is None
