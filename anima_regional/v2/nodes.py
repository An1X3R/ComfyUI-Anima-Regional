"""ComfyUI node definitions for the explicit V2 regional workflow."""

from __future__ import annotations

import copy

import torch

from ..conditioning import encode_prompt, extract_conditioning
from .masks import render_inspection
from .validation import (
    MAX_CHARACTERS,
    _error,
    _text,
    character_id_from_unique_id,
    layout_from_inputs,
    make_character_payload,
    validate_layout,
)


def _validate_pack(pack):
    if not isinstance(pack, dict) or pack.get("version") != 2:
        raise _error("invalid ANIMA_REGIONAL_PACK_V2 input")
    layout = validate_layout(pack.get("layout"))
    entries = pack.get("characters")
    if not isinstance(entries, list) or len(entries) != len(layout["characters"]):
        raise _error("prompt pack character entries do not match its layout")
    expected = [item["uuid"] for item in layout["characters"]]
    actual = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("conditioning"), dict):
            raise _error("prompt pack has an invalid character conditioning entry")
        spec = entry["conditioning"]
        if not torch.is_tensor(spec.get("raw")):
            raise _error("prompt pack character conditioning has no tensor embedding")
        for name in ("ids", "weights"):
            value = spec.get(name)
            if value is not None and not torch.is_tensor(value):
                raise _error(f"prompt pack character conditioning {name} must be a tensor")
        identity_spec = entry.get("identity_conditioning")
        if identity_spec is not None:
            if not isinstance(identity_spec, dict) or not torch.is_tensor(
                identity_spec.get("raw")
            ):
                raise _error(
                    "prompt pack character identity conditioning has no tensor embedding"
                )
            for name in ("ids", "weights"):
                value = identity_spec.get(name)
                if value is not None and not torch.is_tensor(value):
                    raise _error(
                        f"prompt pack character identity conditioning {name} "
                        "must be a tensor"
                    )
        actual.append(entry.get("uuid"))
    if actual != expected or len(set(actual)) != len(actual):
        raise _error("prompt pack character UUID mapping is invalid")
    shared = pack.get("shared_conditioning")
    if shared is not None:
        if not isinstance(shared, dict) or not torch.is_tensor(shared.get("raw")):
            raise _error("prompt pack shared conditioning has no tensor embedding")
        for name in ("ids", "weights"):
            value = shared.get(name)
            if value is not None and not torch.is_tensor(value):
                raise _error(f"prompt pack shared conditioning {name} must be a tensor")
    shared_is_final = pack.get("shared_is_final", False)
    if not isinstance(shared_is_final, bool):
        raise _error("prompt pack shared_is_final must be boolean")
    if "positive" not in pack or "negative" not in pack:
        raise _error("prompt pack has no final positive or negative conditioning")
    for name in ("positive", "negative"):
        conditioning = pack[name]
        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            raise _error(f"prompt pack final {name} conditioning is empty")
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or not item or not torch.is_tensor(item[0]):
                raise _error(f"prompt pack final {name} conditioning is invalid")
    return layout


class AnimaRegionalCharacterPromptV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "Character"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "color": ("STRING", {"default": ""}),
            },
            "optional": {
                # The frontend owns this stable value and passes it back on reload.
                "character_uuid": ("STRING", {"default": ""}),
                "identity_prompt": ("STRING", {"multiline": True, "default": ""}),
                "pose_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("ANIMA_REGIONAL_CHARACTER_V2",)
    RETURN_NAMES = ("character",)
    FUNCTION = "build"
    CATEGORY = "Anima/Regional"

    def build(
        self,
        label,
        prompt,
        strength,
        color,
        character_uuid="",
        identity_prompt="",
        pose_prompt="",
        unique_id=None,
    ):
        if not str(character_uuid or "").strip() and unique_id is not None:
            character_uuid = character_id_from_unique_id(unique_id)
        return (
            make_character_payload(
                label,
                prompt,
                strength,
                color,
                character_uuid,
                identity_prompt,
                pose_prompt,
            ),
        )


class AnimaRegionalLayoutV2:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"character_{index}": ("ANIMA_REGIONAL_CHARACTER_V2",)
            for index in range(1, MAX_CHARACTERS + 1)
        }
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "overlap_mode": (["exclusive", "normalized"], {"default": "exclusive"}),
                "layout_json": ("STRING", {"multiline": True, "default": "{\"version\":2,\"regions\":[]}"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("ANIMA_REGIONAL_LAYOUT_V2", "INT", "INT")
    RETURN_NAMES = ("layout", "width", "height")
    FUNCTION = "build"
    CATEGORY = "Anima/Regional"

    def build(self, width, height, overlap_mode, layout_json, **kwargs):
        supplied = [kwargs.get(f"character_{index}") for index in range(1, MAX_CHARACTERS + 1)]
        layout = layout_from_inputs(width, height, overlap_mode, layout_json, supplied)
        return layout, layout["width"], layout["height"]


class AnimaRegionalPromptPackV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "layout": ("ANIMA_REGIONAL_LAYOUT_V2",),
                "global_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "base_positive": ("CONDITIONING",),
                "base_negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("ANIMA_REGIONAL_PACK_V2",)
    RETURN_NAMES = ("regional_pack",)
    FUNCTION = "pack"
    CATEGORY = "Anima/Regional"

    def pack(self, clip, layout, global_prompt, negative_prompt, base_positive=None, base_negative=None):
        layout = validate_layout(layout)
        global_text = _text(global_prompt, "global_prompt", required=False)
        negative_text = _text(negative_prompt, "negative_prompt", required=False)
        shared_positive = encode_prompt(clip, global_text, "shared scene")
        positive = base_positive if base_positive is not None else shared_positive
        negative = base_negative if base_negative is not None else encode_prompt(clip, negative_text, "negative")

        # Iterate characters, never regions: a UUID is encoded exactly once.
        characters = []
        for character in layout["characters"]:
            identity_text = character.get("identity_prompt", "")
            pose_text = character.get("pose_prompt", "")
            split_prompt = bool(identity_text or pose_text)
            character_parts = (
                [identity_text, pose_text, character["prompt"]]
                if split_prompt
                else [character["prompt"]]
            )
            text = "\n".join(
                value for value in [*character_parts, global_text] if value
            )
            conditioning = encode_prompt(clip, text, f"character {character['label']}")
            entry = {
                "uuid": character["uuid"],
                "label": character["label"],
                "strength": character["strength"],
                "conditioning": extract_conditioning(conditioning, f"character {character['label']}"),
            }
            if identity_text:
                identity_full_text = "\n".join(
                    value for value in [identity_text, global_text] if value
                )
                identity_conditioning = encode_prompt(
                    clip,
                    identity_full_text,
                    f"character {character['label']} identity",
                )
                entry["identity_conditioning"] = extract_conditioning(
                    identity_conditioning,
                    f"character {character['label']} identity",
                )
            characters.append(entry)
        return ({
            "version": 2,
            "layout": copy.deepcopy(layout),
            "characters": characters,
            # The runtime can subtract this common scene response from each
            # character branch, leaving a cleaner identity/clothing residual.
            "shared_conditioning": extract_conditioning(
                shared_positive,
                "shared scene",
            ),
            "shared_is_final": base_positive is None,
            # These are the only final conditioning lists returned by Apply V2.
            "positive": positive,
            "negative": negative,
        },)


class AnimaRegionalSharedPromptV2:
    """Emit one shared scene prompt for all regional and artist branches.

    This is intentionally a plain STRING node.  Keeping encoding in Prompt
    Pack and Artist Pack means the same text can fan out without creating a
    second, potentially mismatched CLIP conditioning branch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scene_prompt",)
    FUNCTION = "emit"
    CATEGORY = "Anima/Regional"

    def emit(self, scene_prompt):
        return (str(scene_prompt or ""),)


def _route_with_compat(model, pack, blend_mode, advanced_options):
    """Late-bind V2 routing; it resolves the shared compat helper internally."""
    try:
        from .runtime import apply_v2_regional
    except ImportError as exc:
        raise _error("V2 routing runtime is unavailable; install the 0.2.0 compatibility component") from exc
    return apply_v2_regional(model, pack, blend_mode, advanced_options or {})


class AnimaRegionalApplyV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regional_pack": ("ANIMA_REGIONAL_PACK_V2",),
                "enabled": ("BOOLEAN", {"default": True}),
                "blend_mode": (["replace", "base_preserve"], {"default": "replace"}),
            },
            "optional": {
                "advanced_options": ("ANIMA_REGIONAL_OPTIONS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "status")
    FUNCTION = "apply"
    CATEGORY = "Anima/Regional"

    def apply(self, model, regional_pack, enabled, blend_mode, advanced_options=None):
        layout = _validate_pack(regional_pack)
        if blend_mode not in ("replace", "base_preserve"):
            raise _error("blend_mode must be replace or base_preserve")
        positive = regional_pack["positive"]
        negative = regional_pack["negative"]
        if not enabled:
            return model, positive, negative, "disabled; original model returned"
        active_characters = {
            item["character_uuid"]
            for item in layout["regions"]
            if item["enabled"]
            and (
                item["type"] == "body_region"
                or layout["overlap_mode"] == "exclusive"
            )
        }
        if not active_characters:
            return model, positive, negative, "active; no enabled body regions, original model returned"
        patched = _route_with_compat(model, regional_pack, blend_mode, advanced_options)
        if patched is None:
            raise _error("V2 routing compatibility helper returned no model")
        options = dict(advanced_options or {})
        composition_mode = str(options.get("composition_mode", "off"))
        multi_character_guard = str(options.get("multi_character_guard", "off"))
        detail_preserve_mode = str(options.get("detail_preserve_mode", "off"))
        identity_anchor_mode = str(options.get("identity_anchor_mode", "off"))
        identity_detail_mode = str(options.get("identity_detail_mode", "off"))
        composition_text = f", composition={composition_mode}"
        if composition_mode == "early_layout":
            composition_text += (
                f" (strength={float(options.get('composition_strength', 1.0)):.2f},"
                f" expand={float(options.get('composition_expand', 0.04)):.2f},"
                f" end={float(options.get('composition_end_percent', 0.55)):.2f})"
            )
        composition_text += f", multi_guard={multi_character_guard}"
        composition_text += f", detail={detail_preserve_mode}"
        composition_text += f", identity_anchor={identity_anchor_mode}"
        composition_text += f", identity_detail={identity_detail_mode}"
        return (
            patched,
            positive,
            negative,
            f"active: {len(active_characters)} characters, "
            f"overlap={layout['overlap_mode']}, blend={blend_mode}{composition_text}",
        )


class AnimaRegionalInspectV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"layout": ("ANIMA_REGIONAL_LAYOUT_V2",)},
            "optional": {"advanced_options": ("ANIMA_REGIONAL_OPTIONS",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "ownership_preview",
        "region_masks",
        "box_masks_image",
        "effective_masks_image",
    )
    FUNCTION = "render"
    CATEGORY = "Anima/Regional"

    def render(self, layout, advanced_options=None):
        options = dict(advanced_options or {})
        return render_inspection(
            validate_layout(layout),
            boundary_falloff=max(0, int(options.get("boundary_falloff", 0))),
        )
