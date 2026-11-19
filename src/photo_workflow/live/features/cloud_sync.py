"""
Live Cloud Sync stage: upload exported images and their derived tiers to Cloudflare R2.

The sync logic follows `storercd/photo-portal` (`sync/sync.py`): the same long-edge tiers
(`thumb` 600 px, `med` 1200 px, `web` 2000 px, plus the untouched `full` file), keys that
embed a content hash so a changed export gets new keys, and an immutable cache policy.
photo-portal's collection filtering is dropped; only a star-rating filter remains, and
images below it are removed from the bucket.

Bucket layout: `photos/<tier>/<photo stem>-<hash>.jpg`.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from PIL import Image

from photo_workflow.live.api import (
    PermanentStageError,
    StageImplementation,
    StageJob,
    StageRegistry,
    StageResult,
)
from photo_workflow.live.bypass import BypassLedger
from photo_workflow.live.rating import read_rating
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import StageId

LOGGER = logging.getLogger(__name__)
SIZES = {"thumb": 600, "med": 1200, "web": 2000}  # long edge in px, as in photo-portal
FULL_TIER = "full"
TIERS = (*SIZES, FULL_TIER)
KEY_PREFIX = "photos"
JPEG_QUALITY = 85
CACHE_CONTROL = "public, max-age=31536000, immutable"
DELETE_BATCH_SIZE = 1000
DIGEST_LENGTH = 16
ClientFactory = Callable[[str, str, str], Any]


@dataclass(frozen=True)
class R2Config:
    """Validated R2 connection settings."""

    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_settings(cls, settings: LiveSettings) -> R2Config:
        """
        Read the R2 fields of the Live settings.

        Returns:
            The configuration.

        Raises:
            PermanentStageError: If a required field is empty (retrying cannot help).
        """
        config = cls(
            settings.r2_account_id.strip(),
            settings.r2_access_key_id.strip(),
            settings.r2_secret_access_key,
            settings.r2_bucket.strip(),
        )
        if not all((config.account_id, config.access_key_id, config.secret_access_key)):
            raise PermanentStageError("R2 account ID, access key ID and secret key are required")
        if not config.bucket:
            raise PermanentStageError("R2 bucket name is required")
        return config


def create_r2_client(account_id: str, access_key_id: str, secret_access_key: str) -> Any:  # noqa: ANN401
    """
    Create an S3 client for a Cloudflare R2 account.

    Returns:
        A boto3 S3 client.
    """
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )


def file_digest(path: Path) -> str:
    """
    Hash a file's content; the digest is part of every key so edits get new URLs.

    Returns:
        A short hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:DIGEST_LENGTH]


def tier_key(tier: str, stem: str, digest: str) -> str:
    """Return the bucket key of one tier of a photo."""
    return f"{KEY_PREFIX}/{tier}/{stem}-{digest}.jpg"


def resized_jpeg(image: Image.Image, long_edge: int) -> bytes:
    """
    Downscale to a long edge (never upscaling) and encode as JPEG.

    Returns:
        The JPEG bytes.
    """
    copy = image.copy()
    copy.thumbnail((long_edge, long_edge))
    buffer = io.BytesIO()
    copy.convert("RGB").save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


class CloudSync(StageImplementation):
    """Keeps the R2 bucket in step with a photo's export and rating."""

    def __init__(self, client_factory: ClientFactory = create_r2_client) -> None:
        """Create the stage; `client_factory(account, key_id, secret)` builds the S3 client."""
        self._client_factory = client_factory
        self._clients: dict[tuple[str, str, str], Any] = {}

    def process(self, job: StageJob) -> StageResult:
        """
        Sync one photo.

        Below the rating filter the photo's objects are deleted from the bucket and the
        stage is skipped. Otherwise missing tiers are generated and uploaded, and tiers
        of an older export (different hash) are deleted.

        Returns:
            A done result, or a skipped one when the rating is below the filter.

        Raises:
            PermanentStageError: If R2 settings are incomplete or no export exists.
        """
        config = R2Config.from_settings(job.settings)
        client = self._client(config)
        stem = job.photo.stem
        if BypassLedger(job.layout).is_sent(stem):
            return StageResult.skipped("a bypass version is already in the cloud")
        existing = self._list_keys(client, config.bucket, stem)
        rating = read_rating(job.photo.with_suffix(".xmp"))
        if rating < job.settings.r2_min_rating:
            self._delete(client, config.bucket, existing)
            return StageResult.skipped(
                f"rated {rating}, below the {job.settings.r2_min_rating}-star filter"
            )
        if job.source_stage is not StageId.EXPORT:
            raise PermanentStageError(f"No exported image to sync for {job.photo.name}")
        uploaded, removed = self._sync_file(client, config.bucket, stem, job.source, existing)
        return StageResult.done(message=f"uploaded {uploaded} files, removed {removed} outdated")

    def send_direct(self, settings: LiveSettings, path: Path) -> str:
        """
        Upload a bypass export under its own name, replacing the auto version of the frame.

        Returns:
            A short description of what was uploaded.
        """
        config = R2Config.from_settings(settings)
        client = self._client(config)
        existing = self._list_keys(client, config.bucket, path.stem)
        uploaded, removed = self._sync_file(client, config.bucket, path.stem, path, existing)
        return f"uploaded {uploaded} files, removed {removed} outdated"

    def _sync_file(
        self, client: Any, bucket: str, stem: str, source: Path, existing: set[str]  # noqa: ANN401
    ) -> tuple[int, int]:
        """
        Upload the missing tiers of `source` under `stem` and delete tiers of older versions.

        Returns:
            The number of files uploaded and the number removed.
        """
        digest = file_digest(source)
        wanted = {tier: tier_key(tier, stem, digest) for tier in TIERS}
        missing = {tier: key for tier, key in wanted.items() if key not in existing}
        self._upload(client, bucket, source, missing)
        stale = existing - set(wanted.values())
        self._delete(client, bucket, stale)
        return len(missing), len(stale)

    def _client(self, config: R2Config) -> Any:  # noqa: ANN401
        """Return a cached client for the credentials."""
        identity = (config.account_id, config.access_key_id, config.secret_access_key)
        if identity not in self._clients:
            self._clients[identity] = self._client_factory(*identity)
        return self._clients[identity]

    @staticmethod
    def _list_keys(client: Any, bucket: str, stem: str) -> set[str]:  # noqa: ANN401
        """
        List the bucket keys that belong to a photo, across all tiers.

        Returns:
            The existing keys, ignoring look-alike stems.
        """
        pattern = re.compile(
            rf"{KEY_PREFIX}/(?:{'|'.join(TIERS)})/{re.escape(stem)}-[0-9a-f]{{{DIGEST_LENGTH}}}\.jpg"
        )
        keys: set[str] = set()
        paginator = client.get_paginator("list_objects_v2")
        for tier in TIERS:
            prefix = f"{KEY_PREFIX}/{tier}/{stem}-"
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                keys.update(
                    item["Key"]
                    for item in page.get("Contents", [])
                    if pattern.fullmatch(item["Key"])
                )
        return keys

    @staticmethod
    def _upload(client: Any, bucket: str, exported: Path, missing: dict[str, str]) -> None:  # noqa: ANN401
        """Generate and upload the missing tiers; the image is decoded only once."""
        if not missing:
            return
        image: Image.Image | None = None
        for tier, key in missing.items():
            extra: dict[str, str] = {}
            if tier == FULL_TIER:
                body = exported.read_bytes()
                extra["ContentDisposition"] = f'attachment; filename="{exported.name}"'
            else:
                if image is None:
                    with Image.open(exported) as opened:
                        opened.load()
                        image = opened.copy()
                body = resized_jpeg(image, SIZES[tier])
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="image/jpeg",
                CacheControl=CACHE_CONTROL,
                **extra,
            )

    @staticmethod
    def _delete(client: Any, bucket: str, keys: set[str]) -> None:  # noqa: ANN401
        """
        Delete keys in batches.

        Raises:
            RuntimeError: If R2 reports keys it could not delete.
        """
        ordered = sorted(keys)
        for start in range(0, len(ordered), DELETE_BATCH_SIZE):
            batch = ordered[start : start + DELETE_BATCH_SIZE]
            response = client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )
            if response.get("Errors"):
                raise RuntimeError(f"R2 could not delete {len(response['Errors'])} objects")


def register(registry: StageRegistry) -> None:
    """Register the Cloud Sync stage."""
    registry.register(StageId.CLOUD_SYNC, CloudSync())
