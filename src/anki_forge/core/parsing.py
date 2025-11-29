"""Shared parsing utilities for LLM responses."""

import json
import re
import logging
from typing import Optional

from .models import ClozeTarget, ClozeTargetType

logger = logging.getLogger(__name__)

# Mapping from string type names to ClozeTargetType enum
TARGET_TYPE_MAP = {
    "KEY_TERM": ClozeTargetType.KEY_TERM,
    "DEFINITION": ClozeTargetType.DEFINITION,
    "FOREIGN": ClozeTargetType.FOREIGN_PHRASE,
    "FOREIGN_PHRASE": ClozeTargetType.FOREIGN_PHRASE,
    "PHRASE": ClozeTargetType.FULL_PHRASE,
    "FULL_PHRASE": ClozeTargetType.FULL_PHRASE,
    "CONCEPT": ClozeTargetType.CONCEPT,
}


def parse_target_from_dict(
    item: dict,
    max_cloze_groups: int = 3,
) -> Optional[ClozeTarget]:
    """Parse a single target dict into a ClozeTarget object.

    Args:
        item: Dictionary with target data (text, type, importance, etc.)
        max_cloze_groups: Maximum allowed cloze group number

    Returns:
        ClozeTarget if valid, None if invalid
    """
    if not isinstance(item, dict):
        return None

    text = item.get("text", "")
    if not text:
        return None

    # Parse type
    type_str = item.get("type", "KEY_TERM").upper()
    target_type = TARGET_TYPE_MAP.get(type_str, ClozeTargetType.KEY_TERM)

    # Parse importance (default 5, range 1-10)
    importance = item.get("importance", 5)
    if isinstance(importance, str):
        try:
            importance = int(importance)
        except ValueError:
            importance = 5
    importance = max(1, min(10, importance))

    # Parse reason and group
    reason = item.get("reason", "")
    group = item.get("group", 1)
    group = max(1, min(max_cloze_groups, group))

    return ClozeTarget(
        text=text,
        target_type=target_type,
        importance=importance,
        reason=reason,
        cloze_group=group,
    )


def parse_targets_from_list(
    target_list: list,
    max_cloze_groups: int = 3,
) -> list[ClozeTarget]:
    """Parse a list of target dicts into ClozeTarget objects.

    Args:
        target_list: List of dictionaries with target data
        max_cloze_groups: Maximum allowed cloze group number

    Returns:
        List of valid ClozeTarget objects
    """
    targets = []
    for item in target_list:
        target = parse_target_from_dict(item, max_cloze_groups)
        if target:
            targets.append(target)
    return targets


def parse_targets_from_json(
    content: str,
    max_cloze_groups: int = 3,
) -> list[ClozeTarget]:
    """Parse cloze targets from JSON string (typically LLM response).

    Args:
        content: String potentially containing JSON array of targets
        max_cloze_groups: Maximum allowed cloze group number

    Returns:
        List of valid ClozeTarget objects
    """
    try:
        # Find JSON array in content
        match = re.search(r'\[[\s\S]*\]', content)
        if not match:
            logger.warning(f"No JSON array found in content (length={len(content)})")
            logger.debug(f"Content preview: {content[:200]}...")
            return []

        data = json.loads(match.group())
        if not isinstance(data, list):
            logger.warning(f"Expected JSON array, got {type(data).__name__}")
            return []

        return parse_targets_from_list(data, max_cloze_groups)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse targets JSON: {e}")
        logger.debug(f"Content preview: {content[:200]}...")
        return []
