"""Action tag import tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from photo_workflow import action_tags

RECORD = {
    "file": "20261019_190910_64_CDS14049-cc41dbd2feb11df4.jpg",
    "action": "pass",
    "primary_player": 8,
    "involved_players": [],
    "observed_players": [2, 6, 12],
    "ball_in_frame": True,
    "confidence_action": 0.6,
    "confidence_primary_player": 0.9,
}

XMP = """<x:xmpmeta xmlns:x='adobe:ns:meta/'>
<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
 <rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'
  xmlns:lr='http://ns.adobe.com/lightroom/1.0/'>
  <dc:subject><rdf:Bag>{items}</rdf:Bag></dc:subject>
  <lr:weightedFlatSubject><rdf:Bag>{items}</rdf:Bag></lr:weightedFlatSubject>
  <lr:hierarchicalSubject><rdf:Bag>{items}</rdf:Bag></lr:hierarchicalSubject>
 </rdf:Description>
</rdf:RDF>
</x:xmpmeta>
"""


def write_xmp(path: Path, tags: list[str]) -> None:
    """Write a minimal XMP sidecar containing the given keywords."""
    items = "".join(f"<rdf:li>{tag}</rdf:li>" for tag in tags)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(XMP.format(items=items))


def test_record_to_tags_pads_skips_nulls_and_expands_lists() -> None:
    """Verify mapping, two-digit padding, and null omission."""
    tags = action_tags.record_to_tags({**RECORD, "involved_players": [9], "primary_player": None})

    assert tags == [
        "Action: Pass",
        "InvolvedPlayer: 09",
        "Player: 02",
        "Player: 06",
        "Player: 12",
        "BallInFrame: True",
    ]


def test_record_to_tags_includes_action_notes() -> None:
    """Verify action notes become an ActionNotes tag and blanks are skipped."""
    assert "ActionNotes: jump serve" in action_tags.record_to_tags(
        {**RECORD, "action_notes": " jump serve "}
    )
    assert not any(t.startswith("ActionNotes") for t in action_tags.record_to_tags(RECORD))


def test_load_records_rejects_unknown_fields(tmp_path: Path) -> None:
    """Verify an unaccounted-for field fails before anything is imported."""
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps([{**RECORD, "mystery": 1}]))

    with pytest.raises(action_tags.ActionTagError, match="mystery"):
        action_tags.load_records(label_path)


def test_base_name_strips_hash_and_extension() -> None:
    """Verify the label file name converts to the camera base name."""
    assert action_tags.base_name_from_label_file(RECORD["file"]) == "20261019_190910_64_CDS14049"
    assert (
        action_tags.base_name_from_label_file("20261019_190910_64_CDS14049-2-cc41dbd2feb11df4.jpg")
        == "20261019_190910_64_CDS14049"
    )
    with pytest.raises(action_tags.ActionTagError):
        action_tags.base_name_from_label_file("bad.jpg")


def test_find_xmp_prefers_local_and_uses_date_folders(tmp_path: Path) -> None:
    """Verify date-folder lookup, suffixed folders, and local-before-archive order."""
    local, archive = tmp_path / "local", tmp_path / "archive"
    name = "20261019_190910_64_CDS14049.xmp"
    write_xmp(archive / "2026/10/20261019-ehs-volleyball" / name, [])
    write_xmp(local / "edits/2026/10/20261019" / name, [])
    assert action_tags.find_xmp(name[:-4], [local, archive]) == (
        archive / "2026/10/20261019-ehs-volleyball" / name
    )

    write_xmp(local / "2026/10/20261019" / name, [])
    assert action_tags.find_xmp(name[:-4], [local, archive]) == local / "2026/10/20261019" / name


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="ExifTool is required")
def test_import_replaces_managed_tags_in_all_sections(tmp_path: Path) -> None:
    """Verify old managed tags are removed everywhere and other tags are kept."""
    xmp_path = tmp_path / "2026/10/20261019/20261019_190910_64_CDS14049.xmp"
    write_xmp(xmp_path, ["Project: EHS", "Action: Dig", "Player: 04", "BallInFrame: False"])
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps([RECORD]))

    summary = action_tags.import_label_file(label_path, [tmp_path])

    assert summary.updated == 1
    assert summary.failures == []
    expected = {
        "Project: EHS",
        "Action: Pass",
        "PrimaryPlayer: 08",
        "Player: 02",
        "Player: 06",
        "Player: 12",
        "BallInFrame: True",
    }
    result = subprocess.run(
        ["exiftool", "-json", "-XMP-dc:Subject", "-XMP-lr:WeightedFlatSubject",
         "-XMP-lr:HierarchicalSubject", str(xmp_path)],
        check=True, capture_output=True, text=True,
    )
    entry = json.loads(result.stdout)[0]
    for key in ("Subject", "WeightedFlatSubject", "HierarchicalSubject"):
        assert set(entry[key]) == expected
        assert len(entry[key]) == len(expected)


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="ExifTool is required")
def test_import_fails_without_project_tag(tmp_path: Path) -> None:
    """Verify an XMP without a Project tag is reported and left untouched."""
    xmp_path = tmp_path / "2026/10/20261019/20261019_190910_64_CDS14049.xmp"
    write_xmp(xmp_path, ["Action: Dig"])
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps([RECORD]))

    summary = action_tags.import_label_file(label_path, [tmp_path])

    assert summary.updated == 0
    assert "Project" in summary.failures[0]
    assert "Action: Dig" in xmp_path.read_text()


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="ExifTool is required")
def test_dry_run_also_fails_without_project_tag(tmp_path: Path) -> None:
    """Verify dry-run reports a missing Project tag without writing."""
    xmp_path = tmp_path / "2026/10/20261019/20261019_190910_64_CDS14049.xmp"
    write_xmp(xmp_path, ["Action: Dig"])
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps([RECORD]))

    summary = action_tags.import_label_file(label_path, [tmp_path], dry_run=True)

    assert summary.updated == 0
    assert "Project" in summary.failures[0]


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="ExifTool is required")
@pytest.mark.parametrize(
    ("winner", "expected"), [("first", "Action: Pass"), ("last", "Action: Dig")]
)
def test_duplicate_records_follow_configured_winner(
    tmp_path: Path, winner: str, expected: str
) -> None:
    """Verify the configured record wins when two records map to one sidecar."""
    xmp_path = tmp_path / "2026/10/20261019/20261019_190910_64_CDS14049.xmp"
    write_xmp(xmp_path, ["Project: EHS"])
    copy = {**RECORD, "file": RECORD["file"].replace("CDS14049-", "CDS14049-2-"), "action": "dig"}
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps([RECORD, copy]))

    action_tags.import_label_file(label_path, [tmp_path], duplicate_winner=winner)

    text = xmp_path.read_text()
    assert expected in text
    assert ("Action: Dig" in text) == (winner == "last")
