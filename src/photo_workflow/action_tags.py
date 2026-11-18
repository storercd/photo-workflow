"""Import photo action label JSON files into XMP sidecar tags."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from photo_workflow.config import DEFAULT_DUPLICATE_RECORD_WINNER, load_config

LOGGER = logging.getLogger(__name__)

# Every tag name this importer owns; all existing values are removed on import.
MANAGED_TAG_NAMES = (
    "Action",
    "ActionNotes",
    "PrimaryPlayer",
    "InvolvedPlayer",
    "Player",
    "BallInFrame",
)
# Record fields we read but deliberately do not import.
IGNORED_RECORD_FIELDS = frozenset({"file", "confidence_action", "confidence_primary_player"})
HANDLED_RECORD_FIELDS = frozenset({
    "action",
    "action_notes",
    "primary_player",
    "involved_players",
    "observed_players",
    "ball_in_frame",
})
KNOWN_RECORD_FIELDS = IGNORED_RECORD_FIELDS | HANDLED_RECORD_FIELDS
REQUIRED_TAG_NAME = "Project"
KEYWORD_FIELDS = ("XMP-dc:Subject", "XMP-lr:WeightedFlatSubject", "XMP-lr:HierarchicalSubject")
# Anything after the CDS frame number (hash, virtual-copy "-2", ...) is not part of the camera name.
BASE_NAME_PATTERN = re.compile(r"^(\d{8}_\d{6}_\d{2}_CDS\d+)")
DATE_PREFIX_PATTERN = re.compile(r"^(\d{4})(\d{2})(\d{2})_")


class ActionTagError(Exception):
    """A label record could not be imported."""


@dataclass(frozen=True)
class ImportSummary:
    """Counts and failures from one label-file import."""

    updated: int = 0
    failures: list[str] = field(default_factory=list)


def base_name_from_label_file(label_file: str) -> str:
    """
    Convert a label `file` value to the camera file base name.

    Returns:
        The camera name up to and including the CDS frame number.

    Raises:
        ActionTagError: If the value does not follow the expected naming pattern.
    """
    match = BASE_NAME_PATTERN.match(Path(label_file).name)
    if match is None:
        raise ActionTagError(f"unrecognized file name: {label_file}")
    return match.group(1)


def find_xmp(base_name: str, search_roots: list[Path]) -> Path | None:
    """
    Locate the XMP sidecar using the date in the name to pick the day folders.

    Roots are searched in order, so the local copy wins over the archive.

    Returns:
        The sidecar path, or None when no root has it.
    """
    match = DATE_PREFIX_PATTERN.match(base_name)
    if match is None:
        return None
    year, month, day = match.groups()
    for root in search_roots:
        month_dir = root / year / month
        if not month_dir.is_dir():
            continue
        for day_dir in sorted(month_dir.glob(f"{year}{month}{day}*")):
            if not day_dir.is_dir():
                continue
            for candidate in sorted(day_dir.rglob(f"{base_name}.xmp")):
                if candidate.is_file():
                    return candidate
    return None


def load_records(label_path: Path) -> list[dict]:
    """
    Read a label file and reject any record field the importer does not account for.

    Returns:
        The label records.

    Raises:
        ActionTagError: If a record contains an unknown field.
    """
    records = json.loads(label_path.read_text())
    for index, record in enumerate(records, start=1):
        unknown = sorted(set(record) - KNOWN_RECORD_FIELDS)
        if unknown:
            raise ActionTagError(
                f"{label_path.name}: record {index} ({record.get('file', '<missing>')}) "
                f"has unhandled field(s): {', '.join(unknown)}"
            )
    return records


def format_number(value: int) -> str:
    """Return a number zero-padded to two digits."""
    return f"{value:02d}"


def record_to_tags(record: dict) -> list[str]:
    """
    Convert one label record to `Name: value` tag strings, skipping nulls.

    Returns:
        The tags in a stable order.
    """
    tags: list[str] = []
    action = record.get("action")
    if action is not None:
        tags.append(f"Action: {str(action).capitalize()}")
    notes = record.get("action_notes")
    if notes is not None and str(notes).strip():
        tags.append(f"ActionNotes: {str(notes).strip()}")
    primary = record.get("primary_player")
    if primary is not None:
        tags.append(f"PrimaryPlayer: {format_number(primary)}")
    tags.extend(
        f"InvolvedPlayer: {format_number(number)}"
        for number in record.get("involved_players") or []
        if number is not None
    )
    tags.extend(
        f"Player: {format_number(number)}"
        for number in record.get("observed_players") or []
        if number is not None
    )
    ball = record.get("ball_in_frame")
    if ball is not None:
        tags.append(f"BallInFrame: {'True' if ball else 'False'}")
    return tags


def is_managed_tag(tag: str) -> bool:
    """Return whether a keyword belongs to a tag this importer owns."""
    name, separator, _ = tag.partition(":")
    return bool(separator) and name.strip() in MANAGED_TAG_NAMES


def read_keywords(xmp_path: Path) -> dict[str, list[str]]:
    """
    Read the keyword lists Lightroom maintains from an XMP file.

    Returns:
        A mapping of ExifTool field name to its current values.
    """
    result = subprocess.run(
        ["exiftool", "-json", *(f"-{name}" for name in KEYWORD_FIELDS), str(xmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    entry = json.loads(result.stdout)[0]
    keywords: dict[str, list[str]] = {}
    for name in KEYWORD_FIELDS:
        value = entry.get(name.split(":", 1)[1], [])
        keywords[name] = [value] if isinstance(value, str) else [str(item) for item in value]
    return keywords


def merge_keywords(existing: list[str], new_tags: list[str]) -> list[str]:
    """
    Drop all managed tags from a keyword list and append the new ones.

    Returns:
        The updated keyword list.
    """
    return [tag for tag in existing if not is_managed_tag(tag)] + new_tags


def write_keywords(xmp_path: Path, keywords: dict[str, list[str]]) -> None:
    """Replace the keyword lists in an XMP file."""
    arguments = [
        f"-{name}={value}" for name, values in keywords.items() for value in values
    ]
    subprocess.run(
        ["exiftool", "-overwrite_original", *arguments, str(xmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def apply_tags(xmp_path: Path, new_tags: list[str], *, dry_run: bool = False) -> None:
    """
    Replace the managed tags in every Lightroom keyword section of an XMP file.

    In dry-run mode the Project check still runs but nothing is written.

    Raises:
        ActionTagError: If the XMP has no Project tag (it has not been saved from Lightroom).
    """
    existing = read_keywords(xmp_path)
    if not any(
        tag.partition(":")[0].strip() == REQUIRED_TAG_NAME
        for tags in existing.values()
        for tag in tags
    ):
        raise ActionTagError(f"no '{REQUIRED_TAG_NAME}:' tag found in {xmp_path}")
    if dry_run:
        return
    write_keywords(
        xmp_path,
        {name: merge_keywords(tags, new_tags) for name, tags in existing.items()},
    )


def import_label_file(
    label_path: Path,
    search_roots: list[Path],
    *,
    dry_run: bool = False,
    duplicate_winner: str = DEFAULT_DUPLICATE_RECORD_WINNER,
) -> ImportSummary:
    """
    Import every record in a label file into its matching XMP sidecar.

    Returns:
        The number of sidecars updated and a description of each failure.

    Raises:
        ActionTagError: If the label file contains an unknown record field.
    """
    records = load_records(label_path)
    updated = 0
    failures: list[str] = []
    seen: dict[Path, str] = {}
    for record in records:
        label_file = record.get("file", "<missing>")
        try:
            base_name = base_name_from_label_file(label_file)
            xmp_path = find_xmp(base_name, search_roots)
            if xmp_path is None:
                raise ActionTagError(f"no .xmp found for {base_name}")
            if xmp_path in seen:
                LOGGER.warning(
                    "WARNING: %s and %s both map to %s; the %s record wins",
                    seen[xmp_path],
                    label_file,
                    xmp_path.name,
                    duplicate_winner,
                )
                if duplicate_winner == "first":
                    continue
            seen[xmp_path] = label_file
            new_tags = record_to_tags(record)
            apply_tags(xmp_path, new_tags, dry_run=dry_run)
            if dry_run:
                LOGGER.info("would tag %s: %s", xmp_path, new_tags)
            updated += 1
        except (ActionTagError, subprocess.CalledProcessError) as error:
            failures.append(f"{label_file}: {error}")
    return ImportSummary(updated=updated, failures=failures)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse action tag import command-line arguments.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        prog="photo-workflow-import-action-tags",
        description="Import photo action label JSON files into XMP sidecar tags.",
    )
    parser.add_argument("label_files", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the action tag import command."""
    parsed_args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config()
    search_roots = [config.workflow.camera_root, config.workflow.archive_root]
    try:
        for label_path in parsed_args.label_files:
            load_records(label_path)
    except ActionTagError as error:
        LOGGER.error("%s", error)
        sys.exit(2)
    failed = False
    for label_path in parsed_args.label_files:
        summary = import_label_file(
            label_path,
            search_roots,
            dry_run=parsed_args.dry_run,
            duplicate_winner=config.action_tags.duplicate_record_winner,
        )
        LOGGER.info(
            "%s: %s sidecar(s) %s, %s failure(s)",
            label_path.name,
            summary.updated,
            "matched" if parsed_args.dry_run else "updated",
            len(summary.failures),
        )
        for failure in summary.failures:
            LOGGER.error("  %s", failure)
        failed = failed or bool(summary.failures)
    if failed:
        sys.exit(1)
