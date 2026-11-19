"""
Bypass: starred frames skip the automatic pipeline's result and go out as your own edit.

A photo rated at or above the bypass threshold is copied (RAW and sidecar) to
`bypass-input`. Editing it there is optional. Whatever lands in `bypass-output` (for example
a Lightroom export) is sent straight to the cloud under the same name, replacing the
automatic version, and the automatic Cloud Sync never replaces it again.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from photo_workflow.live.layout import RAW_SUFFIXES, LiveLayout
from photo_workflow.live.rating import read_rating

LOGGER = logging.getLogger(__name__)
LEDGER_NAME = ".bypass-sent.json"
_LEDGER_LOCK = threading.Lock()
Signature = tuple[int, int]
OUTPUT_SUFFIXES = frozenset({".jpg", ".jpeg"})


def file_signature(path: Path) -> Signature:
    """Return a file's size and modification time in nanoseconds, or `(-1, 0)` if unreadable."""
    try:
        stat = path.stat()
    except OSError:
        return (-1, 0)
    return (stat.st_size, stat.st_mtime_ns)


class BypassLedger:
    """Remembers which bypass exports were sent, so the auto version never replaces them."""

    def __init__(self, layout: LiveLayout) -> None:
        """Use the ledger file in the Live folder."""
        self._path = layout.root / LEDGER_NAME

    def _read(self) -> dict[str, list[int]]:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def is_sent(self, stem: str) -> bool:
        """Return whether a bypass version of this frame was sent."""
        with _LEDGER_LOCK:
            return stem in self._read()

    def is_current(self, path: Path) -> bool:
        """Return whether this exact file (same name, size and time) was already sent."""
        with _LEDGER_LOCK:
            return self._read().get(path.stem) == list(file_signature(path))

    def mark_sent(self, path: Path, signature: Signature | None = None) -> None:
        """Record a file as sent; `signature` is the one taken before it was uploaded."""
        with _LEDGER_LOCK:
            data = self._read()
            data[path.stem] = list(signature or file_signature(path))
            self._path.parent.mkdir(parents=True, exist_ok=True)
            partial = self._path.with_name(self._path.name + ".part")
            partial.write_text(json.dumps(data))
            os.replace(partial, self._path)

    def count(self) -> int:
        """Return how many frames went out through the bypass."""
        with _LEDGER_LOCK:
            return len(self._read())


@dataclass
class BypassRouter:
    """Stages starred photos in `bypass-input` and finds finished exports to send."""

    layout: LiveLayout
    _ratings: dict[Path, tuple[float, int]] = field(default_factory=dict)
    _seen: dict[Path, Signature] = field(default_factory=dict)
    _staged_once: set[Path] = field(default_factory=set)

    @property
    def ledger(self) -> BypassLedger:
        """The record of what was sent."""
        return BypassLedger(self.layout)

    def stage_inputs(self, photos: list[Path], threshold: int) -> None:
        """
        Copy photos rated at least `threshold` to `bypass-input`, and withdraw ones that fell below.

        With a threshold of 0 the bypass is off and nothing is changed. Photos whose bypass
        version already went out are left alone.
        """
        if threshold <= 0:
            return
        sent = self.ledger
        for photo in photos:
            staged = self.layout.bypass_input / photo.name
            if self._rating(photo) >= threshold:
                if (
                    not staged.exists()
                    and photo not in self._staged_once
                    and not sent.is_sent(photo.stem)
                ):
                    self._stage(photo, staged)
                    self._staged_once.add(photo)
            else:
                self._staged_once.discard(photo)
                if staged.exists() and not sent.is_sent(photo.stem):
                    self._withdraw(staged)

    def _rating(self, photo: Path) -> int:
        sidecar = photo.with_suffix(".xmp")
        try:
            modified = sidecar.stat().st_mtime
        except OSError:
            return 0
        cached = self._ratings.get(sidecar)
        if cached is None or cached[0] != modified:
            cached = (modified, read_rating(sidecar))
            self._ratings[sidecar] = cached
        return cached[1]

    def _stage(self, photo: Path, staged: Path) -> None:
        try:
            staged.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(photo, staged)
            except OSError:
                shutil.copy2(photo, staged)
            sidecar = photo.with_suffix(".xmp")
            if sidecar.is_file():
                shutil.copy2(sidecar, staged.with_suffix(".xmp"))
        except OSError:
            LOGGER.exception("could not stage %s for bypass", photo.name)
            return
        LOGGER.info("bypass: staged %s", photo.name)

    @staticmethod
    def _withdraw(staged: Path) -> None:
        staged.unlink(missing_ok=True)
        staged.with_suffix(".xmp").unlink(missing_ok=True)
        LOGGER.info("bypass: withdrew %s", staged.name)

    def waiting(self) -> int:
        """Return how many photos are in `bypass-input`."""
        folder = self.layout.bypass_input
        if not folder.is_dir():
            return 0
        return sum(1 for path in folder.iterdir() if path.suffix.lower() in RAW_SUFFIXES)

    def sent(self) -> int:
        """Return how many frames went out through `bypass-output`."""
        return self.ledger.count()

    def ready_outputs(self) -> list[Path]:
        """
        Return finished exports in `bypass-output` that were not sent yet.

        A file is ready once its size and time are unchanged since the previous call, so
        exports still being written are not picked up.

        Returns:
            The files to send.
        """
        folder = self.layout.bypass_output
        if not folder.is_dir():
            self._seen = {}
            return []
        ledger = self.ledger
        current = {
            path: file_signature(path)
            for path in sorted(folder.iterdir())
            if path.is_file()
            and path.suffix.lower() in OUTPUT_SUFFIXES
            and not path.name.startswith(".")
        }
        ready = [
            path
            for path, signature in current.items()
            if self._seen.get(path) == signature and not ledger.is_current(path)
        ]
        self._seen = current
        return ready
