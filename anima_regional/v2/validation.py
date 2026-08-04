"""Strict, deterministic validation for the public V2 payloads."""

from __future__ import annotations

import copy
import json
import math
import re
import uuid as uuid_module
from typing import Any

MAX_CHARACTERS = 8
MAX_REGIONS = 64
MAX_DIMENSION = 8192
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _error(message: str) -> ValueError:
    return ValueError(f"[AnimaRegional V2] {message}")


def _text(value: Any, name: str, *, required: bool = True, limit: int = 4096) -> str:
    if not isinstance(value, str):
        raise _error(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise _error(f"{name} cannot be empty")
    if len(result) > limit:
        raise _error(f"{name} is too long")
    return result


def _stable_id(value: Any, name: str) -> str:
    result = _text(value, name, limit=128)
    if not _STABLE_ID.fullmatch(result):
        raise _error(f"{name} must be a stable identifier")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _error(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise _error(f"{name} must be finite")
    return result


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise _error(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{name} must be an integer") from exc
    if result < 64 or result > MAX_DIMENSION:
        raise _error(f"{name} must be between 64 and {MAX_DIMENSION}")
    return result


def make_character_payload(
    label: Any,
    prompt: Any,
    strength: Any = 1.0,
    color: Any = "",
    character_uuid: Any = "",
    identity_prompt: Any = "",
    pose_prompt: Any = "",
) -> dict[str, Any]:
    """Create a contract-shaped character; supplied IDs remain stable."""
    if character_uuid is not None and not isinstance(character_uuid, str):
        raise _error("character uuid must be a string")
    identifier = character_uuid.strip() if character_uuid is not None else ""
    if not identifier:
        identifier = str(uuid_module.uuid4())
    payload: dict[str, Any] = {
        "version": 2,
        "kind": "character",
        "uuid": identifier,
        "label": label,
        "prompt": prompt,
        "strength": strength,
        "identity_prompt": identity_prompt,
        "pose_prompt": pose_prompt,
    }
    if color is not None and str(color).strip():
        payload["color"] = color
    return validate_character(payload)


def character_id_from_unique_id(unique_id: Any) -> str:
    """Derive a stable character ID from ComfyUI's persistent node ID."""
    value = str(unique_id).strip()
    if not value:
        raise _error("character unique_id cannot be empty")
    safe = re.sub(r"[^A-Za-z0-9._:-]+", "-", value).strip("-._:")
    if not safe:
        raise _error("character unique_id cannot form a stable identifier")
    return _stable_id(f"character-node-{safe}"[:128], "character uuid")


def validate_character(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _error("character payload must be an object")
    if payload.get("version") != 2 or payload.get("kind") != "character":
        raise _error("invalid ANIMA_REGIONAL_CHARACTER_V2 payload")
    prompt = _text(payload.get("prompt", ""), "character prompt", required=False)
    identity_prompt = _text(
        payload.get("identity_prompt", ""),
        "character identity_prompt",
        required=False,
    )
    pose_prompt = _text(
        payload.get("pose_prompt", ""),
        "character pose_prompt",
        required=False,
    )
    if not prompt and not identity_prompt and not pose_prompt:
        raise _error(
            "character requires prompt, identity_prompt, or pose_prompt"
        )
    result: dict[str, Any] = {
        "version": 2,
        "kind": "character",
        "uuid": _stable_id(payload.get("uuid"), "character uuid"),
        "label": _text(payload.get("label"), "character label", limit=256),
        "prompt": prompt,
        "identity_prompt": identity_prompt,
        "pose_prompt": pose_prompt,
        "strength": _number(payload.get("strength"), "character strength"),
    }
    if not 0.0 <= result["strength"] <= 4.0:
        raise _error("character strength must be between 0 and 4")
    color = payload.get("color")
    if color is not None and str(color).strip():
        if not isinstance(color, str) or not _COLOR.fullmatch(color):
            raise _error("character color must be #RRGGBB")
        result["color"] = color.upper()
    return result


def _parse_layout_json(layout_json: Any) -> dict[str, Any]:
    if isinstance(layout_json, dict):
        return copy.deepcopy(layout_json)
    if not isinstance(layout_json, str):
        raise _error("layout_json must be a JSON object")
    text = layout_json.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _error(f"layout_json is invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise _error("layout_json must describe an object")
    return value


def _clamp_coordinate(value: Any, name: str) -> float:
    return min(1.0, max(0.0, _number(value, name)))


def _validate_region(payload: Any, character_ids: set[str], index: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _error(f"region {index} must be an object")
    region_type = payload.get("type")
    if region_type not in ("body_region", "ownership_hint"):
        raise _error(f"region {index} has an invalid type")
    if payload.get("geometry", "box") != "box":
        raise _error(f"region {index} only supports box geometry")
    character_uuid = _stable_id(payload.get("character_uuid"), f"region {index} character_uuid")
    if character_uuid not in character_ids:
        raise _error(f"region {index} references an unknown character")
    width = _number(payload.get("width"), f"region {index} width")
    height = _number(payload.get("height"), f"region {index} height")
    if width <= 0.0 or height <= 0.0:
        raise _error(f"region {index} width and height must be positive")
    x = _clamp_coordinate(payload.get("x"), f"region {index} x")
    y = _clamp_coordinate(payload.get("y"), f"region {index} y")
    # Geometry remains in bounds even when a stale editor payload overshoots.
    width = min(1.0 - x, width)
    height = min(1.0 - y, height)
    if width <= 0.0 or height <= 0.0:
        raise _error(f"region {index} lies outside the canvas")
    feather = min(1.0, max(0.0, _number(payload.get("feather", 0.0), f"region {index} feather")))
    enabled = payload.get("enabled", True)
    if not isinstance(enabled, bool):
        raise _error(f"region {index} enabled must be boolean")
    priority_value = _number(
        payload.get("priority", 0),
        f"region {index} priority",
    )
    priority = int(priority_value)
    if float(priority) != float(priority_value):
        raise _error(f"region {index} priority must be an integer")
    if not -10 <= priority <= 10:
        raise _error(f"region {index} priority must be between -10 and 10")
    result = {
        "uuid": _stable_id(payload.get("uuid"), f"region {index} uuid"),
        "character_uuid": character_uuid,
        "type": region_type,
        "geometry": "box",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "feather": feather,
        "priority": priority,
        "enabled": enabled,
    }
    if region_type == "ownership_hint":
        hint_blend = str(payload.get("hint_blend", "hard") or "hard")
        if hint_blend not in ("hard", "soft"):
            raise _error(
                f"region {index} hint_blend must be hard or soft"
            )
        strength = _number(
            payload.get("strength", 1.0),
            f"region {index} strength",
        )
        if not 0.0 <= strength <= 1.0:
            raise _error(f"region {index} strength must be between 0 and 1")
        result.update({
            "hint_blend": hint_blend,
            "strength": strength,
        })
    return result


def validate_layout(
    layout: Any,
    *,
    width: Any | None = None,
    height: Any | None = None,
    overlap_mode: Any | None = None,
    supplied_characters: list[Any] | None = None,
) -> dict[str, Any]:
    """Return a canonical V2 layout with no editor-owned values trusted."""
    if not isinstance(layout, dict):
        raise _error("layout payload must be an object")
    if layout.get("version") not in (None, 2):
        raise _error("layout version must be 2")
    raw_characters = supplied_characters if supplied_characters is not None else layout.get("characters", [])
    if not isinstance(raw_characters, list) or not raw_characters:
        raise _error("layout requires at least one character")
    if len(raw_characters) > MAX_CHARACTERS:
        raise _error(f"layout supports at most {MAX_CHARACTERS} characters")
    characters = [validate_character(value) for value in raw_characters]
    identifiers = [value["uuid"] for value in characters]
    if len(set(identifiers)) != len(identifiers):
        raise _error("layout contains duplicate character UUIDs")
    raw_regions = layout.get("regions", [])
    if not isinstance(raw_regions, list):
        raise _error("layout regions must be a list")
    if len(raw_regions) > MAX_REGIONS:
        raise _error(f"layout supports at most {MAX_REGIONS} regions")
    regions = [_validate_region(value, set(identifiers), index + 1) for index, value in enumerate(raw_regions)]
    region_ids = [value["uuid"] for value in regions]
    if len(set(region_ids)) != len(region_ids):
        raise _error("layout contains duplicate region UUIDs")
    result = {
        "version": 2,
        "width": _dimension(layout.get("width") if width is None else width, "width"),
        "height": _dimension(layout.get("height") if height is None else height, "height"),
        "overlap_mode": layout.get("overlap_mode") if overlap_mode is None else overlap_mode,
        "characters": characters,
        "regions": regions,
    }
    if result["overlap_mode"] not in ("exclusive", "normalized"):
        raise _error("overlap_mode must be exclusive or normalized")
    return result


def layout_from_inputs(
    width: Any,
    height: Any,
    overlap_mode: Any,
    layout_json: Any,
    character_inputs: list[Any],
) -> dict[str, Any]:
    editor_layout = _parse_layout_json(layout_json)
    supplied = [validate_character(value) for value in character_inputs if value is not None]
    if supplied:
        supplied_ids = [value["uuid"] for value in supplied]
        if len(set(supplied_ids)) != len(supplied_ids):
            raise _error("character inputs contain duplicate UUIDs")
        supplied_by_id = {value["uuid"]: value for value in supplied}
        layout_characters = editor_layout.get("characters")
        if layout_characters is not None:
            editor_ids = [validate_character(value)["uuid"] for value in layout_characters]
            if len(set(editor_ids)) != len(editor_ids):
                raise _error("layout_json contains duplicate character UUIDs")
            if set(editor_ids) != set(supplied_ids):
                raise _error("layout_json character bindings do not match character inputs")
            # Socket order is a UI detail. Keep the editor's UUID order while
            # refreshing each payload from the corresponding upstream node.
            supplied = [supplied_by_id[identifier] for identifier in editor_ids]
        return validate_layout(
            editor_layout,
            width=width,
            height=height,
            overlap_mode=overlap_mode,
            supplied_characters=supplied,
        )
    return validate_layout(editor_layout, width=width, height=height, overlap_mode=overlap_mode)
