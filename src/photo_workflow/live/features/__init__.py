"""
Live stage implementations, one module per feature.

Each module defines `register(registry: StageRegistry) -> None`; every module in this
package is discovered and registered automatically at start-up.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from photo_workflow.live.api import StageRegistry

LOGGER = logging.getLogger(__name__)


def load_features(registry: StageRegistry) -> None:
    """Import every feature module here and let it register its stage implementations."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{module_info.name}")
            module.register(registry)
        except Exception:
            LOGGER.exception("live feature %s failed to load; it stays disabled", module_info.name)
