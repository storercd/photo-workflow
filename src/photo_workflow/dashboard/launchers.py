"""Launching and icon resolution for dashboard application shortcuts."""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib import resources
from pathlib import Path

from photo_workflow.config import DashboardAppLauncher

LOGGER = logging.getLogger(__name__)
APPLICATION_RESOLUTION_TIMEOUT_SECONDS = 2
APPLICATION_LAUNCH_TIMEOUT_SECONDS = 5

# Maps a console-script executable name to the package bundling its dashboard icon
# (expected at <package>/resources/icon.png). Add an entry here for any new
# in-house tool that should show its own icon instead of the generic fallback.
BUNDLED_COMMAND_ICON_PACKAGES = {
    "photo-workflow-crop": "photo_workflow.crop_tool",
}


class LaunchError(RuntimeError):
    """Raised when a dashboard application shortcut fails to launch."""


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


def resolve_bundled_command_icon(executable: str) -> Path | None:
    """Return the bundled dashboard icon for one of Photo Workflow's own console scripts."""
    package = BUNDLED_COMMAND_ICON_PACKAGES.get(executable)
    if package is None:
        return None

    icon_resource = resources.files(package) / "resources" / "icon.png"
    return Path(str(icon_resource)) if icon_resource.is_file() else None


def first_file_in_folder(folder: Path) -> Path | None:
    """Return the first (alphabetically) non-hidden file directly inside a folder."""
    if not folder.is_dir():
        return None
    candidates = sorted(
        entry for entry in folder.iterdir() if entry.is_file() and not entry.name.startswith(".")
    )
    return candidates[0] if candidates else None


def launch_app(launcher: DashboardAppLauncher, target_folder: Path | None = None) -> None:
    """
    Launch a dashboard application shortcut, optionally pointed at a target folder.

    May propagate LaunchError if the underlying launch command reports a failure.
    """
    effective_folder = target_folder if launcher.supports_target_folder else None

    if launcher.command is not None:
        launch_command(launcher.command, effective_folder)
        return

    launch_named_application(launcher.app_name, effective_folder)


def launch_named_application(app_name: str, target_folder: Path | None = None) -> None:
    """
    Launch a macOS application by name using Launch Services.

    May propagate LaunchError if Launch Services reports it could not open the application.
    """
    args = ["open", "-a", app_name]
    if target_folder is not None:
        target_file = first_file_in_folder(target_folder)
        if target_file is not None:
            args.append(str(target_file))
        else:
            LOGGER.warning(
                "no file found in %s; opening %s without a target file", target_folder, app_name
            )

    LOGGER.info("launching application %s", app_name)
    _run_launch(args, description=app_name)


def launch_command(command: tuple[str, ...], target_folder: Path | None = None) -> None:
    """
    Launch an explicit command, resolving Photo Workflow's own console scripts.

    May propagate LaunchError if the command could not be started.
    """
    executable, *arguments = command
    if executable == "photo-workflow-crop":
        args = [sys.executable, "-m", "photo_workflow.crop_tool.app", *arguments]
        if target_folder is not None:
            args.append(str(target_folder))
        LOGGER.info("launching command %s", " ".join(args))
        _popen_launch(args, description="Photo Workflow Crop")
        return

    args = list(command)
    if target_folder is not None:
        args.append(str(target_folder))
    LOGGER.info("launching command %s", " ".join(args))
    _popen_launch(args, description=executable)


def _run_launch(args: list[str], *, description: str) -> None:
    """
    Run a quick dispatching command (e.g. `open`) and raise if it reports failure.

    Raises:
        LaunchError: If the command exits non-zero, times out, or cannot be started.
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=APPLICATION_LAUNCH_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        raise LaunchError(f"could not launch {description}: {error}") from error

    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise LaunchError(f"could not launch {description}: {detail}")


def _popen_launch(args: list[str], *, description: str) -> None:
    """
    Start a long-running process without blocking the UI.

    Raises:
        LaunchError: If the process could not be started.
    """
    try:
        subprocess.Popen(args)
    except OSError as error:
        raise LaunchError(f"could not launch {description}: {error}") from error
