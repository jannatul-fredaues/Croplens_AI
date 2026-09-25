"""
Loads the static crop-care lookup table (api/data/crop_info.json).

Kept as plain, sourced JSON rather than model-generated text, so the
agronomy advice shown to users is consistent and checkable. See
Day 4 of the build plan.
"""

import json
import logging
from pathlib import Path

from app.config import get_settings
from app.schemas import CareInfo

logger = logging.getLogger("croplens.knowledge")

_care_data: dict[str, CareInfo] = {}


def load_knowledge() -> None:
    """Load crop_info.json into memory. Call once at startup."""
    settings = get_settings()
    path = Path(settings.crop_info_path)

    if not path.exists():
        # Non-fatal: predictions still work, just without a care section.
        # This is expected until Day 4, when data/crop_info.json is filled in.
        logger.warning("crop_info.json not found at %s - care info will be empty", path)
        return

    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    for label, fields in raw.items():
        if label.startswith("_"):
            continue  # metadata keys like "_readme", not a class
        try:
            _care_data[label] = CareInfo(**fields)
        except Exception:
            logger.exception("Invalid crop_info.json entry for label '%s'", label)

    logger.info("Loaded care info for %d classes", len(_care_data))


def get_care_info(label: str) -> CareInfo | None:
    """Return care info for a predicted class label, or None if unknown."""
    return _care_data.get(label)


def is_loaded() -> bool:
    return bool(_care_data)
