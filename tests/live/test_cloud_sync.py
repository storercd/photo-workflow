"""Tests for the Live Cloud Sync stage; the R2 client is faked, nothing touches the network."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from photo_workflow.live.api import PermanentStageError, StageJob, StageRegistry
from photo_workflow.live.features import cloud_sync
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import StageId, StageStatus

STEM = "20251031_101500_00_IMG_0001"
XMP_TEMPLATE = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{rating}"/>'
    "</rdf:RDF></x:xmpmeta>"
)


ELEMENT_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description>'
    '<xmp:Rating xmlns:xmp="http://ns.adobe.com/xap/1.0/">2</xmp:Rating>'
    "</rdf:Description></rdf:RDF></x:xmpmeta>"
)


class FakeR2:
    """In-memory stand-in for the boto3 S3 client."""

    def __init__(self) -> None:
        """Start with an empty bucket."""
        self.objects: dict[str, dict[str, Any]] = {}
        self.put_calls: list[str] = []
        self.fail_puts = 0

    def put_object(self, **kwargs: Any) -> None:
        """
        Store an object, failing first if asked to.

        Raises:
            ConnectionError: While `fail_puts` is positive.
        """
        if self.fail_puts:
            self.fail_puts -= 1
            raise ConnectionError("network down")
        self.objects[kwargs["Key"]] = kwargs
        self.put_calls.append(kwargs["Key"])

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:  # noqa: N803
        """
        Remove objects.

        Returns:
            An empty response.
        """
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}

    def get_paginator(self, name: str) -> FakeR2:
        """
        Return this object as the only supported paginator.

        Returns:
            The fake itself.
        """
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:  # noqa: N803
        """
        List matching keys as a single page.

        Returns:
            One response page.
        """
        return [
            {"Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)]}
        ]


@pytest.fixture
def r2() -> FakeR2:
    """Return a fresh fake bucket."""
    return FakeR2()


@pytest.fixture
def stage(r2: FakeR2) -> cloud_sync.CloudSync:
    """Return Cloud Sync wired to the fake bucket."""
    return cloud_sync.CloudSync(lambda account, key, secret: r2)


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """
    Create a photo with an exported JPEG and an XMP sidecar rated 4.

    Returns:
        The path of the original.
    """
    date_dir = tmp_path / "live" / "2025" / "10" / "20251031"
    (date_dir / "input").mkdir(parents=True)
    (date_dir / "export").mkdir()
    original = date_dir / "input" / f"{STEM}.CR3"
    original.write_bytes(b"raw")
    original.with_suffix(".xmp").write_text(XMP_TEMPLATE.format(rating=4))
    write_export(date_dir / "export" / f"{STEM}.jpg", "red")
    return original


def write_export(path: Path, color: str) -> None:
    """Write a 3000x2000 JPEG export."""
    buffer = io.BytesIO()
    Image.new("RGB", (3000, 2000), color).save(buffer, "JPEG")
    path.write_bytes(buffer.getvalue())


def make_job(photo: Path, min_rating: int = 3, **overrides: Any) -> StageJob:
    """
    Build a Cloud Sync job whose source is the photo's export.

    Returns:
        The job.
    """
    settings = LiveSettings(
        r2_account_id="acct",
        r2_access_key_id="key",
        r2_secret_access_key="secret",
        r2_bucket="bucket",
        r2_min_rating=min_rating,
    )
    fields: dict[str, Any] = {
        "stage": StageId.CLOUD_SYNC,
        "photo": photo,
        "source": photo.parent.parent / "export" / f"{STEM}.jpg",
        "source_stage": StageId.EXPORT,
        "layout": LiveLayout(photo.parents[4]),
        "settings": settings,
        "warn": lambda message: None,
    }
    fields.update(overrides)
    return StageJob(**fields)


def keys_by_tier(r2: FakeR2) -> dict[str, str]:
    """
    Group stored keys by their tier folder.

    Returns:
        A tier to key mapping.
    """
    return {key.split("/")[1]: key for key in r2.objects}


def test_uploads_every_tier_with_a_content_hash_key(
    stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path
) -> None:
    """Uploads every tier with a content hash key."""
    result = stage.process(make_job(photo))

    keys = keys_by_tier(r2)
    assert result.status is StageStatus.DONE
    assert set(keys) == {"thumb", "med", "web", "full"}
    digest = cloud_sync.file_digest(photo.parent.parent / "export" / f"{STEM}.jpg")
    assert all(key == f"photos/{tier}/{STEM}-{digest}.jpg" for tier, key in keys.items())
    for tier, long_edge in cloud_sync.SIZES.items():
        with Image.open(io.BytesIO(r2.objects[keys[tier]]["Body"])) as tier_image:
            assert max(tier_image.size) == min(long_edge, 3000)
    full = r2.objects[keys["full"]]
    assert full["Body"] == (photo.parent.parent / "export" / f"{STEM}.jpg").read_bytes()
    assert "attachment" in full["ContentDisposition"]
    assert full["CacheControl"] == cloud_sync.CACHE_CONTROL


def test_second_run_uploads_nothing(stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path) -> None:
    """Second run uploads nothing."""
    stage.process(make_job(photo))
    r2.put_calls.clear()

    stage.process(make_job(photo))

    assert r2.put_calls == []


def test_missing_tier_is_regenerated(stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path) -> None:
    """Missing tier is regenerated."""
    stage.process(make_job(photo))
    thumb = keys_by_tier(r2)["thumb"]
    del r2.objects[thumb]
    r2.put_calls.clear()

    stage.process(make_job(photo))

    assert r2.put_calls == [thumb]


def test_changed_export_uploads_new_keys_and_deletes_old_ones(
    stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path
) -> None:
    """Changed export uploads new keys and deletes old ones."""
    stage.process(make_job(photo))
    old_keys = set(r2.objects)
    write_export(photo.parent.parent / "export" / f"{STEM}.jpg", "green")

    stage.process(make_job(photo))

    assert len(r2.objects) == 4
    assert not old_keys & set(r2.objects)


def test_below_the_rating_filter_deletes_the_photo_from_the_bucket(
    stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path
) -> None:
    """Below the rating filter deletes the photo from the bucket."""
    stage.process(make_job(photo, min_rating=3))
    other = "photos/thumb/20251031_101500_00_IMG_00011-0123456789abcdef.jpg"
    r2.objects[other] = {}

    result = stage.process(make_job(photo, min_rating=5))

    assert result.status is StageStatus.SKIPPED
    assert set(r2.objects) == {other}  # only the look-alike stem survives


def test_unrated_photo_is_below_a_filter_and_eligible_without_one(
    stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path
) -> None:
    """Unrated photo is below a filter and eligible without one."""
    photo.with_suffix(".xmp").unlink()

    assert stage.process(make_job(photo, min_rating=1)).status is StageStatus.SKIPPED
    assert r2.objects == {}
    assert stage.process(make_job(photo, min_rating=0)).status is StageStatus.DONE


def test_incomplete_settings_halt_immediately(stage: cloud_sync.CloudSync, photo: Path) -> None:
    """Incomplete settings halt immediately."""
    job = make_job(photo, settings=LiveSettings())

    with pytest.raises(PermanentStageError):
        stage.process(job)


def test_requires_an_export_source(stage: cloud_sync.CloudSync, photo: Path) -> None:
    """Requires an export source."""
    job = make_job(photo, source=photo, source_stage=None)

    with pytest.raises(PermanentStageError):
        stage.process(job)


def test_upload_errors_propagate_so_the_framework_can_retry(
    stage: cloud_sync.CloudSync, r2: FakeR2, photo: Path
) -> None:
    """Upload errors propagate so the framework can retry."""
    r2.fail_puts = 1

    with pytest.raises(ConnectionError):
        stage.process(make_job(photo))

    assert stage.process(make_job(photo)).status is StageStatus.DONE


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (XMP_TEMPLATE.format(rating=5), 5),
        (XMP_TEMPLATE.format(rating=-1), 0),
        ("not xmp", 0),
        (ELEMENT_XMP, 2),
    ],
)
def test_read_rating(tmp_path: Path, body: str, expected: int) -> None:
    """Read rating."""
    sidecar = tmp_path / "a.xmp"
    sidecar.write_text(body)

    assert cloud_sync.read_rating(sidecar) == expected


def test_register_provides_the_cloud_sync_stage() -> None:
    """Register provides the cloud sync stage."""
    registry = StageRegistry()
    cloud_sync.register(registry)

    assert registry.is_registered(StageId.CLOUD_SYNC)
