"""Launcher resolution and launching tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from photo_workflow.config import DashboardAppLauncher
from photo_workflow.dashboard import launchers


def test_resolve_bundled_command_icon_returns_path_for_known_command() -> None:
    """Verify the crop tool's own command resolves to its bundled icon file."""
    icon_path = launchers.resolve_bundled_command_icon("photo-workflow-crop")

    assert icon_path is not None
    assert icon_path.name == "icon.png"
    assert icon_path.is_file()


def test_resolve_bundled_command_icon_returns_none_for_unknown_command() -> None:
    """Verify commands without a bundled icon mapping return None."""
    assert launchers.resolve_bundled_command_icon("some-other-tool") is None


def test_resolve_application_path_returns_path_for_existing_app(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify a resolvable application name returns its bundle path."""
    app_path = tmp_path / "Aftershoot.app"
    app_path.mkdir()

    monkeypatch.setattr(
        launchers.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=f"{app_path}\n", stderr=""
        ),
    )

    resolved = launchers.resolve_application_path("Aftershoot")

    assert resolved == app_path


def test_resolve_application_path_returns_none_when_resolution_fails(monkeypatch) -> None:
    """Verify a failed AppleScript lookup returns None instead of raising."""

    def raise_called_process_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(launchers.subprocess, "run", raise_called_process_error)

    assert launchers.resolve_application_path("Nonexistent App") is None


def test_resolve_application_path_returns_none_when_path_does_not_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a resolved but missing bundle path returns None."""
    missing_path = tmp_path / "Missing.app"

    monkeypatch.setattr(
        launchers.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=f"{missing_path}\n", stderr=""
        ),
    )

    assert launchers.resolve_application_path("Missing") is None


def test_launch_app_opens_named_application(monkeypatch) -> None:
    """Verify an app_name launcher is opened via macOS Launch Services."""
    run_calls: list[list[str]] = []
    monkeypatch.setattr(
        launchers.subprocess,
        "run",
        lambda command, **kwargs: (
            run_calls.append(command)
            or subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        ),
    )

    launchers.launch_app(DashboardAppLauncher(name="Aftershoot", app_name="Aftershoot"))

    assert run_calls == [["open", "-a", "Aftershoot"]]


def test_launch_app_opens_named_application_with_first_file_in_target_folder(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify a target-folder-capable app is opened with its first file."""
    (tmp_path / "b.jpg").write_text("b")
    (tmp_path / "a.jpg").write_text("a")
    run_calls: list[list[str]] = []
    monkeypatch.setattr(
        launchers.subprocess,
        "run",
        lambda command, **kwargs: (
            run_calls.append(command)
            or subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        ),
    )

    launchers.launch_app(
        DashboardAppLauncher(
            name="FastRawViewer", app_name="FastRawViewer", supports_target_folder=True
        ),
        target_folder=tmp_path,
    )

    assert run_calls == [["open", "-a", "FastRawViewer", str(tmp_path / "a.jpg")]]


def test_launch_app_ignores_target_folder_when_unsupported(tmp_path: Path, monkeypatch) -> None:
    """Verify a launcher that does not opt in ignores the target folder."""
    (tmp_path / "a.jpg").write_text("a")
    run_calls: list[list[str]] = []
    monkeypatch.setattr(
        launchers.subprocess,
        "run",
        lambda command, **kwargs: (
            run_calls.append(command)
            or subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        ),
    )

    launchers.launch_app(
        DashboardAppLauncher(name="Aftershoot", app_name="Aftershoot"), target_folder=tmp_path
    )

    assert run_calls == [["open", "-a", "Aftershoot"]]


def test_launch_app_raises_launch_error_on_failure(monkeypatch) -> None:
    """Verify a non-zero Launch Services exit is surfaced as a LaunchError."""
    monkeypatch.setattr(
        launchers.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, stdout="", stderr="No such application"
        ),
    )

    with pytest.raises(launchers.LaunchError, match="Ghost App.*No such application"):
        launchers.launch_app(DashboardAppLauncher(name="Ghost App", app_name="Ghost App"))


def test_launch_app_runs_photo_workflow_crop_as_a_module(monkeypatch) -> None:
    """Verify the crop tool is launched via -m so it uses this interpreter's venv."""
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(launchers.subprocess, "Popen", lambda command: popen_calls.append(command))

    launchers.launch_app(
        DashboardAppLauncher(name="Photo Workflow Crop", command=("photo-workflow-crop",))
    )

    assert popen_calls == [[sys.executable, "-m", "photo_workflow.crop_tool.app"]]


def test_launch_app_passes_target_folder_to_photo_workflow_crop(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify the crop tool receives the target folder as a positional argument."""
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(launchers.subprocess, "Popen", lambda command: popen_calls.append(command))

    launchers.launch_app(
        DashboardAppLauncher(
            name="Photo Workflow Crop",
            command=("photo-workflow-crop",),
            supports_target_folder=True,
        ),
        target_folder=tmp_path,
    )

    assert popen_calls == [[sys.executable, "-m", "photo_workflow.crop_tool.app", str(tmp_path)]]


def test_launch_app_runs_other_commands_directly(monkeypatch) -> None:
    """Verify a non-crop command launcher runs the command as given."""
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(launchers.subprocess, "Popen", lambda command: popen_calls.append(command))

    launchers.launch_app(
        DashboardAppLauncher(name="Custom Tool", command=("custom-tool", "--flag"))
    )

    assert popen_calls == [["custom-tool", "--flag"]]
