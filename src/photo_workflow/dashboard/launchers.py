"""Launching and icon resolution for dashboard application shortcuts."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from photo_workflow.config import DashboardAppLauncher

LOGGER = logging.getLogger(__name__)
APPLICATION_RESOLUTION_TIMEOUT_SECONDS = 2


def resolve_application_path(app_name: str) -> Path | None:
    """Return the installed .app bundle path for a named application, if found."""
    try:
        result = subprocess.run(
            ["osascript", "-e", f'POSIX path of (path to application "{app_name}")'],
            capture_output=True,
            text=True,
            timeout=APPLICATION_RESOLUTION_TIMEOUT_SECONDS,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None

    resolved_path = Path(result.stdout.strip())
    return resolved_path if resolved_path.exists() else None


def launch_app(launcher: DashboardAppLauncher) -> None:
    """Launch a dashboard application shortcut without blocking the UI."""
    if launcher.command is not None:
        launch_command(launcher.command)
        return

    launch_named_application(launcher.app_name)


def launch_named_application(app_name: str) -> None:
    """Launch a macOS application by name using Launch Services."""
    LOGGER.info("launching application %s", app_name)
    subprocess.Popen(["open", "-a", app_name])


def launch_command(command: tuple[str, ...]) -> None:
    """Launch an explicit command, resolving Photo Workflow's own console scripts."""
    LOGGER.info("launching command %s", " ".join(command))
    executable, *arguments = command
    if executable == "photo-workflow-crop":
        subprocess.Popen([sys.executable, "-m", "photo_workflow.crop_tool.app", *arguments])
        return

    subprocess.Popen(list(command))
