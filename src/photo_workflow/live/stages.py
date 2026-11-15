"""Stage definitions for the Live pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StageId(StrEnum):
    """Identifiers of the pipeline stages, in execution order."""

    FTP = "ftp"
    DENOISE = "denoise"
    MASK = "mask"
    LIGHTING = "lighting"
    AUTOCROP = "autocrop"
    EXPORT = "export"
    CLOUD_SYNC = "cloud_sync"


class StageStatus(StrEnum):
    """Status of one image within one stage."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class StageSpec:
    """
    Static description of a stage.

    Attributes:
        id: The stage identifier.
        label: Human readable name shown in the status panel.
        subfolders: Output subfolders of the date folder; the first is the primary one.
        blocks_when_disabled: When true, images wait at this stage while its feature is
            off. When false, a disabled stage is passed through and later stages fall
            back to the most advanced earlier source.
        concurrency: Maximum images processed at once by this stage.
    """

    id: StageId
    label: str
    subfolders: tuple[str, ...] = ()
    blocks_when_disabled: bool = False
    concurrency: int = 1


STAGES: tuple[StageSpec, ...] = (
    StageSpec(StageId.FTP, "FTP", subfolders=("input",)),
    StageSpec(StageId.DENOISE, "Denoise", subfolders=("denoise",)),
    StageSpec(StageId.MASK, "Mask", subfolders=("mask",)),
    StageSpec(StageId.LIGHTING, "Lighting/Color", subfolders=("lighting", "output")),
    StageSpec(StageId.AUTOCROP, "AutoCrop"),
    StageSpec(StageId.EXPORT, "Export", subfolders=("export",), blocks_when_disabled=True),
    StageSpec(StageId.CLOUD_SYNC, "Cloud Sync", blocks_when_disabled=True),
)
SPECS: dict[StageId, StageSpec] = {spec.id: spec for spec in STAGES}
STAGE_ORDER: tuple[StageId, ...] = tuple(spec.id for spec in STAGES)
INPUT_SUBFOLDER = "input"
OUTPUT_SUBFOLDER = "output"


def downstream_of(stage: StageId) -> tuple[StageId, ...]:
    """Return the stages that come after `stage`."""
    return STAGE_ORDER[STAGE_ORDER.index(stage) + 1 :]


def upstream_of(stage: StageId) -> tuple[StageId, ...]:
    """Return the stages that come before `stage`, nearest first."""
    return tuple(reversed(STAGE_ORDER[: STAGE_ORDER.index(stage)]))
