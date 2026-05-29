from __future__ import annotations

from pathlib import Path

from photo_workflow import config as app_config
from photo_workflow import workflow


def test_main_runs_memory_card_import_before_video_notes(tmp_path: Path, monkeypatch) -> None:
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
        lambda target_dir, config: call_order.append("import"),
    )
    monkeypatch.setattr(
        workflow,
        "run_video_notes_step",
        lambda source_dir, config: call_order.append("video_notes"),
    )

    workflow.main()

    assert call_order == ["import", "video_notes"]
