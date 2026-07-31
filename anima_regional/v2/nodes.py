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

_V2_MULTI_CHARACTER_GUARDS = ("off", "soft", "strong")
_V2_DETAIL_PRESERVE_MODES = ("off", "soft", "strong")
_V2_IDENTITY_ANCHOR_MODES = ("off", "shared_delta")


def _validate_pack(pack):
    if not isinstance(pack, dict) or pack.get("version") != 2:
        raise _error("invalid ANIMA_REGIONAL_PACK_V2 input")
    layout = validate_layout(pack.get("layout"))
    entries = pack.get("characters")
    if not isinstance(entries, list) or len(entries) != len(layout["characters"]):
        raise _error("prompt pack character entries do not match its layout")
    expected = [item["uuid"] for item in layout["characters"]]
    actual = []
    def validate_spec(spec, label, *, required=False):
        if spec is None and not required:
            return
        if not isinstance(spec, dict) or not torch.is_tensor(spec.get("raw")):
            raise _error(f"prompt pack {label} conditioning has no tensor embedding")
        for name in ("ids", "weights"):
            value = spec.get(name)
            if value is not None and not torch.is_tensor(value):
                raise _error(
                    f"prompt pack {label} conditioning {name} must be a tensor"
                )

    routing_contract = pack.get("routing_contract", "legacy_v2")
    if routing_contract not in ("legacy_v2", "separated_v1"):
        raise _error("prompt pack has an unsupported routing contract")
    for entry in entries:
        if not isinstance(entry, dict):
            raise _error("prompt pack has an invalid character conditioning entry")
        validate_spec(entry.get("conditioning"), "character", required=True)
        validate_spec(entry.get("identity_conditioning"), "character identity")
        validate_spec(entry.get("pose_conditioning"), "character pose")
        actual.append(entry.get("uuid"))
    if actual != expected or len(set(actual)) != len(actual):
        raise _error("prompt pack character UUID mapping is invalid")
    validate_spec(pack.get("shared_conditioning"), "shared")
    validate_spec(pack.get("background_conditioning"), "background")
    validate_spec(pack.get("layout_conditioning"), "layout")
    if routing_contract == "separated_v1" and not isinstance(
        pack.get("background_conditioning"), dict
    ):
        raise _error("separated prompt pack requires background conditioning")
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
                "layout_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("ANIMA_REGIONAL_PACK_V2",)
    RETURN_NAMES = ("regional_pack",)
    FUNCTION = "pack"
    CATEGORY = "Anima/Regional"

    def pack(
        self,
        clip,
        layout,
        global_prompt,
        negative_prompt,
        base_positive=None,
        base_negative=None,
        layout_prompt="",
    ):
        layout = validate_layout(layout)
        background_text = _text(global_prompt, "global_prompt", required=False)
        layout_text = _text(layout_prompt, "layout_prompt", required=False)
        negative_text = _text(negative_prompt, "negative_prompt", required=False)
        background_positive = encode_prompt(clip, background_text, "background/style")
        background_spec = extract_conditioning(
            background_positive,
            "background/style",
        )
        positive = base_positive if base_positive is not None else background_positive
        negative = base_negative if base_negative is not None else encode_prompt(clip, negative_text, "negative")

        layout_spec = None
        if layout_text:
            layout_spec = extract_conditioning(
                encode_prompt(clip, layout_text, "group layout/presence"),
                "group layout/presence",
            )

        # Iterate characters, never regions. Stable identity and pose are
        # encoded separately so group count/layout text is never copied into
        # every identity branch.
        characters = []
        warnings = []
        for character in layout["characters"]:
            identity_text = character.get("identity_prompt", "")
            pose_text = character.get("pose_prompt", "")
            legacy_text = character.get("prompt", "")
            legacy_ignored = bool(identity_text and legacy_text)
            if not identity_text:
                identity_text = legacy_text
            elif legacy_ignored:
                warnings.append(
                    f"{character['label']}: ignored legacy prompt because identity_prompt is set"
                )

            identity_spec = None
            if identity_text:
                identity_spec = extract_conditioning(
                    encode_prompt(
                        clip,
                        identity_text,
                        f"character {character['label']} identity",
                    ),
                    f"character {character['label']} identity",
                )

            pose_spec = None
            if pose_text:
                pose_spec = extract_conditioning(
                    encode_prompt(
                        clip,
                        pose_text,
                        f"character {character['label']} pose",
                    ),
                    f"character {character['label']} pose",
                )

            # ``conditioning`` remains as a compatibility alias for older
            # runtimes and third-party pack inspectors. New routing consumes
            # the explicit identity/pose fields below.
            compatibility_spec = identity_spec or pose_spec
            if compatibility_spec is None:
                raise _error(
                    f"character {character['label']} has no identity or pose conditioning"
                )
            entry = {
                "uuid": character["uuid"],
                "label": character["label"],
                "strength": character["strength"],
                "conditioning": compatibility_spec,
                "legacy_prompt_ignored": legacy_ignored,
            }
            if identity_spec is not None:
                entry["identity_conditioning"] = identity_spec
            if pose_spec is not None:
                entry["pose_conditioning"] = pose_spec
            characters.append(entry)
        return ({
            "version": 2,
            "routing_contract": "separated_v1",
            "layout": copy.deepcopy(layout),
            "characters": characters,
            "background_conditioning": background_spec,
            "background_is_final": base_positive is None,
            "layout_conditioning": layout_spec,
            # Backward-compatible aliases for 0.3.x tooling.
            "shared_conditioning": background_spec,
            "shared_is_final": base_positive is None,
            "warnings": warnings,
            # These are the only final conditioning lists returned by Apply V2.
            "positive": positive,
            "negative": negative,
        },)


class AnimaRegionalSharedPromptV2:
    """Emit one background/style prompt for all regional and artist branches.

    This is intentionally a plain STRING node. Keeping encoding in Prompt Pack
    and Artist Pack means the same background/style text can fan out without
    creating a second, potentially mismatched CLIP conditioning branch.
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


class AnimaRegionalOptionsV2:
    """Advanced controls whose values are consumed by the V2 router."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "start_block": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
                "end_block": ("INT", {"default": -1, "min": -1, "max": 63, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "boundary_falloff": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "composition_mode": (["off", "early_layout"], {"default": "off"}),
                "composition_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "composition_expand": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.15, "step": 0.01}),
                "composition_end_percent": ("FLOAT", {"default": 0.55, "min": 0.05, "max": 1.0, "step": 0.01}),
                "layout_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "multi_character_guard": (["off", "soft", "strong"], {"default": "off"}),
                "detail_preserve_mode": (["off", "soft", "strong"], {"default": "off"}),
                "detail_preserve_start": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 0.95, "step": 0.05}),
                "detail_preserve_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05}),
                "edge_focus_power": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "identity_anchor_mode": (["off", "shared_delta"], {"default": "off"}),
                "identity_anchor_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "identity_late_floor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    # Reuse the public socket type so this node can connect to Apply V2 while
    # remaining separate from the legacy Advanced Options UI.
    RETURN_TYPES = ("ANIMA_REGIONAL_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "Anima/Regional/V2 Advanced"

    def build(
        self,
        global_strength,
        start_block,
        end_block,
        start_percent,
        end_percent,
        boundary_falloff=0,
        composition_mode="off",
        composition_strength=1.0,
        composition_expand=0.04,
        composition_end_percent=0.55,
        layout_strength=1.0,
        multi_character_guard="off",
        detail_preserve_mode="off",
        detail_preserve_start=0.65,
        detail_preserve_amount=0.5,
        edge_focus_power=1.0,
        identity_anchor_mode="off",
        identity_anchor_strength=1.0,
        identity_late_floor=0.8,
    ):
        if float(start_percent) > float(end_percent):
            raise _error("start_percent must not exceed end_percent")
        if not 0.0 <= float(global_strength) <= 2.0:
            raise _error("global_strength must be between 0 and 2")
        if str(composition_mode) not in ("off", "early_layout"):
            raise _error("composition_mode must be off or early_layout")
        if not 0.0 <= float(composition_strength) <= 2.0:
            raise _error("composition_strength must be between 0 and 2")
        if not 0.0 <= float(composition_expand) <= 0.15:
            raise _error("composition_expand must be between 0 and 0.15")
        if not 0.05 <= float(composition_end_percent) <= 1.0:
            raise _error("composition_end_percent must be between 0.05 and 1")
        if not 0.0 <= float(layout_strength) <= 2.0:
            raise _error("layout_strength must be between 0 and 2")
        if str(multi_character_guard) not in _V2_MULTI_CHARACTER_GUARDS:
            raise _error("multi_character_guard must be off, soft, or strong")
        if str(detail_preserve_mode) not in _V2_DETAIL_PRESERVE_MODES:
            raise _error("detail_preserve_mode must be off, soft, or strong")
        if not 0.0 <= float(detail_preserve_start) <= 0.95:
            raise _error("detail_preserve_start must be between 0 and 0.95")
        if not 0.0 <= float(detail_preserve_amount) <= 0.95:
            raise _error("detail_preserve_amount must be between 0 and 0.95")
        if not 1.0 <= float(edge_focus_power) <= 4.0:
            raise _error("edge_focus_power must be between 1 and 4")
        if str(identity_anchor_mode) not in _V2_IDENTITY_ANCHOR_MODES:
            raise _error("identity_anchor_mode must be off or shared_delta")
        if not 0.0 <= float(identity_anchor_strength) <= 1.0:
            raise _error("identity_anchor_strength must be between 0 and 1")
        if not 0.0 <= float(identity_late_floor) <= 1.0:
            raise _error("identity_late_floor must be between 0 and 1")
        return ({
            "global_strength": float(global_strength),
            "start_block": int(start_block),
            "end_block": int(end_block),
            "start_percent": float(start_percent),
            "end_percent": float(end_percent),
            "boundary_falloff": max(0, int(boundary_falloff)),
            "composition_mode": str(composition_mode),
            "composition_strength": float(composition_strength),
            "composition_expand": float(composition_expand),
            "composition_end_percent": float(composition_end_percent),
            "layout_strength": float(layout_strength),
            "multi_character_guard": str(multi_character_guard),
            "detail_preserve_mode": str(detail_preserve_mode),
            "detail_preserve_start": float(detail_preserve_start),
            "detail_preserve_amount": float(detail_preserve_amount),
            "edge_focus_power": float(edge_focus_power),
            "identity_anchor_mode": str(identity_anchor_mode),
            "identity_anchor_strength": float(identity_anchor_strength),
            "identity_late_floor": float(identity_late_floor),
            "options_contract": "v2_routing_v1",
        },)


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
        routing_contract = regional_pack.get("routing_contract", "legacy_v2")
        layout_strength = float(options.get("layout_strength", 1.0))
        composition_text = f", composition={composition_mode}"
        if composition_mode == "early_layout":
            composition_text += (
                f" (strength={float(options.get('composition_strength', 1.0)):.2f},"
                f" expand={float(options.get('composition_expand', 0.04)):.2f},"
                f" end={float(options.get('composition_end_percent', 0.55)):.2f})"
            )
        composition_text += f", multi_guard={multi_character_guard}"
        composition_text += f", detail={detail_preserve_mode}"
        composition_text += f", layout_strength={layout_strength:.2f}"
        if routing_contract == "separated_v1":
            composition_text += ", route=background+layout+pose+identity"
        else:
            composition_text += f", identity_anchor={identity_anchor_mode}"
            composition_text += f", identity_detail={identity_detail_mode}"
        warning_text = ""
        if regional_pack.get("warnings"):
            warning_text = "; warnings=" + " | ".join(regional_pack["warnings"])
        return (
            patched,
            positive,
            negative,
            f"active: {len(active_characters)} characters, "
            f"overlap={layout['overlap_mode']}, blend={blend_mode}{composition_text}{warning_text}",
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
