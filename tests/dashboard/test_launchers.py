"""Launcher resolution and launching tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from photo_workflow.config import DashboardAppLauncher
from photo_workflow.dashboard import launchers


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
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(
        launchers.subprocess, "Popen", lambda command: popen_calls.append(command)
    )

    launchers.launch_app(DashboardAppLauncher(name="Aftershoot", app_name="Aftershoot"))

    assert popen_calls == [["open", "-a", "Aftershoot"]]


def test_launch_app_runs_photo_workflow_crop_as_a_module(monkeypatch) -> None:
    """Verify the crop tool is launched via -m so it uses this interpreter's venv."""
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(
        launchers.subprocess, "Popen", lambda command: popen_calls.append(command)
    )

    launchers.launch_app(
        DashboardAppLauncher(name="Photo Workflow Crop", command=("photo-workflow-crop",))
    )

    assert popen_calls == [[sys.executable, "-m", "photo_workflow.crop_tool.app"]]


def test_launch_app_runs_other_commands_directly(monkeypatch) -> None:
    """Verify a non-crop command launcher runs the command as given."""
    popen_calls: list[list[str]] = []
    monkeypatch.setattr(
        launchers.subprocess, "Popen", lambda command: popen_calls.append(command)
    )

    launchers.launch_app(
        DashboardAppLauncher(name="Custom Tool", command=("custom-tool", "--flag"))
    )

    assert popen_calls == [["custom-tool", "--flag"]]
