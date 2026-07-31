"""ComfyUI node definitions for regional Anima prompting."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .conditioning import encode_prompt, extract_conditioning
from .masks import build_raw_region_weights, build_region_weights
from .runtime import (
    get_diffusion_model,
    make_cross_attention_patch,
    make_runtime_capture,
    make_self_attention_patch,
    validate_anima,
)

logger = logging.getLogger(__name__)

MAX_REGIONS = 8


def _region_items(regions):
    if regions is None:
        return []
    if not isinstance(regions, dict) or regions.get("version") != 1:
        raise ValueError("[AnimaRegional] invalid ANIMA_REGIONS input")
    items = regions.get("items")
    if not isinstance(items, list):
        raise ValueError("[AnimaRegional] region chain has no item list")
    return list(items)


def _append_region(regions, item):
    items = _region_items(regions)
    if len(items) >= MAX_REGIONS:
        raise ValueError(f"[AnimaRegional] at most {MAX_REGIONS} regions are supported")
    items.append(item)
    return {"version": 1, "items": items}


class AnimaRegionalCharacterBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "center_x": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "center_y": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "width": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.01, "max": 2.0, "step": 0.01},
                ),
                "height": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.01},
                ),
                "feather": (
                    "FLOAT",
                    {"default": 0.04, "min": 0.0, "max": 0.25, "step": 0.005},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "priority": (
                    "INT",
                    {"default": 0, "min": -10, "max": 10, "step": 1},
                ),
            },
            "optional": {"regions": ("ANIMA_REGIONS",)},
        }

    RETURN_TYPES = ("ANIMA_REGIONS",)
    RETURN_NAMES = ("regions",)
    FUNCTION = "append"
    CATEGORY = "Anima/Regional/Legacy"

    def append(
        self,
        character_prompt,
        center_x,
        center_y,
        width,
        height,
        feather,
        strength,
        priority,
        regions=None,
    ):
        prompt = str(character_prompt).strip()
        if not prompt:
            raise ValueError("[AnimaRegional] character_prompt cannot be empty")
        item = {
            "geometry": "box",
            "prompt": prompt,
            "center_x": float(center_x),
            "center_y": float(center_y),
            "width": float(width),
            "height": float(height),
            "feather": float(feather),
            "strength": float(strength),
            "priority": int(priority),
        }
        return (_append_region(regions, item),)


class AnimaRegionalCharacterMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "mask": ("MASK",),
                "anchor_x": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "anchor_y": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "priority": (
                    "INT",
                    {"default": 0, "min": -10, "max": 10, "step": 1},
                ),
            },
            "optional": {"regions": ("ANIMA_REGIONS",)},
        }

    RETURN_TYPES = ("ANIMA_REGIONS",)
    RETURN_NAMES = ("regions",)
    FUNCTION = "append"
    CATEGORY = "Anima/Regional/Legacy"

    def append(
        self,
        character_prompt,
        mask,
        anchor_x,
        anchor_y,
        strength,
        priority,
        regions=None,
    ):
        prompt = str(character_prompt).strip()
        if not prompt:
            raise ValueError("[AnimaRegional] character_prompt cannot be empty")
        item = {
            "geometry": "mask",
            "prompt": prompt,
            "mask": mask,
            "anchor_x": float(anchor_x),
            "anchor_y": float(anchor_y),
            "strength": float(strength),
            "priority": int(priority),
        }
        return (_append_region(regions, item),)


class AnimaRegionalOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlap_mode": (
                    ["exclusive", "normalized"],
                    {"default": "exclusive"},
                ),
                "blend_mode": (
                    ["replace", "base_preserve"],
                    {"default": "replace"},
                ),
                "self_attention_mode": (
                    ["off", "strict_experimental"],
                    {"default": "off"},
                ),
                "global_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "start_block": (
                    "INT",
                    {"default": 0, "min": 0, "max": 63, "step": 1},
                ),
                "end_block": (
                    "INT",
                    {"default": -1, "min": -1, "max": 63, "step": 1},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "branch_chunk_size": (
                    "INT",
                    {"default": 2, "min": 1, "max": MAX_REGIONS, "step": 1},
                ),
                "self_attn_start_block": (
                    "INT",
                    {"default": 8, "min": 0, "max": 63, "step": 1},
                ),
                "self_attn_end_block": (
                    "INT",
                    {"default": -1, "min": -1, "max": 63, "step": 1},
                ),
                "self_attn_max_tokens": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 16384, "step": 256},
                ),
            },
            "optional": {
                "boundary_falloff": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8, "step": 1},
                ),
                "composition_mode": (
                    ["off", "early_layout"],
                    {"default": "off"},
                ),
                "composition_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "composition_expand": (
                    "FLOAT",
                    {"default": 0.04, "min": 0.0, "max": 0.15, "step": 0.01},
                ),
                "composition_end_percent": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.05, "max": 1.0, "step": 0.01},
                ),
                "multi_character_guard": (
                    ["off", "soft", "strong"],
                    {"default": "off"},
                ),
                "detail_preserve_mode": (
                    ["off", "soft", "strong"],
                    {"default": "off"},
                ),
                "detail_preserve_start": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 0.95, "step": 0.05},
                ),
                "detail_preserve_amount": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05},
                ),
                "edge_focus_power": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1},
                ),
                "identity_anchor_mode": (
                    ["off", "shared_delta"],
                    {"default": "off"},
                ),
                "identity_anchor_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "identity_late_floor": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "identity_detail_mode": (
                    ["off", "late"],
                    {"default": "off"},
                ),
                "identity_detail_start": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05},
                ),
                "identity_detail_strength": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "identity_core_strength": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.05},
                ),
                "identity_core_radius": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.05, "max": 1.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("ANIMA_REGIONAL_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "Anima/Regional/Advanced"

    def build(
        self,
        overlap_mode,
        blend_mode,
        self_attention_mode,
        global_strength,
        start_block,
        end_block,
        start_percent,
        end_percent,
        branch_chunk_size,
        self_attn_start_block,
        self_attn_end_block,
        self_attn_max_tokens,
        boundary_falloff=0,
        composition_mode="off",
        composition_strength=1.0,
        composition_expand=0.04,
        composition_end_percent=0.55,
        multi_character_guard="off",
        detail_preserve_mode="off",
        detail_preserve_start=0.65,
        detail_preserve_amount=0.5,
        edge_focus_power=1.0,
        identity_anchor_mode="off",
        identity_anchor_strength=1.0,
        identity_late_floor=0.8,
        identity_detail_mode="off",
        identity_detail_start=0.6,
        identity_detail_strength=0.65,
        identity_core_strength=1.5,
        identity_core_radius=0.55,
    ):
        if float(start_percent) > float(end_percent):
            raise ValueError("[AnimaRegional] start_percent must not exceed end_percent")
        composition_mode = str(composition_mode)
        if composition_mode not in ("off", "early_layout"):
            raise ValueError(
                "[AnimaRegional] composition_mode must be off or early_layout"
            )
        composition_strength = float(composition_strength)
        if not 0.0 <= composition_strength <= 2.0:
            raise ValueError(
                "[AnimaRegional] composition_strength must be between 0 and 2"
            )
        composition_expand = float(composition_expand)
        if not 0.0 <= composition_expand <= 0.15:
            raise ValueError(
                "[AnimaRegional] composition_expand must be between 0 and 0.15"
            )
        composition_end_percent = float(composition_end_percent)
        if not 0.05 <= composition_end_percent <= 1.0:
            raise ValueError(
                "[AnimaRegional] composition_end_percent must be between 0.05 and 1"
            )
        multi_character_guard = str(multi_character_guard)
        if multi_character_guard not in ("off", "soft", "strong"):
            raise ValueError(
                "[AnimaRegional] multi_character_guard must be off, soft, or strong"
            )
        detail_preserve_mode = str(detail_preserve_mode)
        if detail_preserve_mode not in ("off", "soft", "strong"):
            raise ValueError(
                "[AnimaRegional] detail_preserve_mode must be off, soft, or strong"
            )
        detail_preserve_start = float(detail_preserve_start)
        if not 0.0 <= detail_preserve_start <= 0.95:
            raise ValueError(
                "[AnimaRegional] detail_preserve_start must be between 0 and 0.95"
            )
        detail_preserve_amount = float(detail_preserve_amount)
        if not 0.0 <= detail_preserve_amount <= 0.95:
            raise ValueError(
                "[AnimaRegional] detail_preserve_amount must be between 0 and 0.95"
            )
        edge_focus_power = float(edge_focus_power)
        if not 1.0 <= edge_focus_power <= 4.0:
            raise ValueError(
                "[AnimaRegional] edge_focus_power must be between 1 and 4"
            )
        identity_anchor_mode = str(identity_anchor_mode)
        if identity_anchor_mode not in ("off", "shared_delta"):
            raise ValueError(
                "[AnimaRegional] identity_anchor_mode must be off or shared_delta"
            )
        identity_anchor_strength = float(identity_anchor_strength)
        if not 0.0 <= identity_anchor_strength <= 1.0:
            raise ValueError(
                "[AnimaRegional] identity_anchor_strength must be between 0 and 1"
            )
        identity_late_floor = float(identity_late_floor)
        if not 0.0 <= identity_late_floor <= 1.0:
            raise ValueError(
                "[AnimaRegional] identity_late_floor must be between 0 and 1"
            )
        identity_detail_mode = str(identity_detail_mode)
        if identity_detail_mode not in ("off", "late"):
            raise ValueError(
                "[AnimaRegional] identity_detail_mode must be off or late"
            )
        identity_detail_start = float(identity_detail_start)
        if not 0.0 <= identity_detail_start <= 0.95:
            raise ValueError(
                "[AnimaRegional] identity_detail_start must be between 0 and 0.95"
            )
        identity_detail_strength = float(identity_detail_strength)
        if not 0.0 <= identity_detail_strength <= 2.0:
            raise ValueError(
                "[AnimaRegional] identity_detail_strength must be between 0 and 2"
            )
        identity_core_strength = float(identity_core_strength)
        if not 1.0 <= identity_core_strength <= 3.0:
            raise ValueError(
                "[AnimaRegional] identity_core_strength must be between 1 and 3"
            )
        identity_core_radius = float(identity_core_radius)
        if not 0.05 <= identity_core_radius <= 1.0:
            raise ValueError(
                "[AnimaRegional] identity_core_radius must be between 0.05 and 1"
            )
        return (
            {
                "overlap_mode": str(overlap_mode),
                "blend_mode": str(blend_mode),
                "self_attention_mode": str(self_attention_mode),
                "global_strength": float(global_strength),
                "start_block": int(start_block),
                "end_block": int(end_block),
                "start_percent": float(start_percent),
                "end_percent": float(end_percent),
                "branch_chunk_size": int(branch_chunk_size),
                "self_attn_start_block": int(self_attn_start_block),
                "self_attn_end_block": int(self_attn_end_block),
                "self_attn_max_tokens": int(self_attn_max_tokens),
                "boundary_falloff": max(0, int(boundary_falloff)),
                "composition_mode": composition_mode,
                "composition_strength": composition_strength,
                "composition_expand": composition_expand,
                "composition_end_percent": composition_end_percent,
                "multi_character_guard": multi_character_guard,
                "detail_preserve_mode": detail_preserve_mode,
                "detail_preserve_start": detail_preserve_start,
                "detail_preserve_amount": detail_preserve_amount,
                "edge_focus_power": edge_focus_power,
                "identity_anchor_mode": identity_anchor_mode,
                "identity_anchor_strength": identity_anchor_strength,
                "identity_late_floor": identity_late_floor,
                "identity_detail_mode": identity_detail_mode,
                "identity_detail_start": identity_detail_start,
                "identity_detail_strength": identity_detail_strength,
                "identity_core_strength": identity_core_strength,
                "identity_core_radius": identity_core_radius,
            },
        )


def _resolve_sigma_range(model, start_percent: float, end_percent: float):
    if start_percent <= 0.0 and end_percent >= 1.0:
        return None
    try:
        sampling = model.get_model_object("model_sampling")
        start_sigma = float(sampling.percent_to_sigma(start_percent))
        end_sigma = float(sampling.percent_to_sigma(end_percent))
        return tuple(sorted((start_sigma, end_sigma)))
    except Exception as exc:
        raise ValueError(
            f"[AnimaRegional] failed to resolve sampling range: {exc}"
        ) from exc


class AnimaRegionalApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "regions": ("ANIMA_REGIONS",),
                "global_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {"options": ("ANIMA_REGIONAL_OPTIONS",)},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "status")
    FUNCTION = "apply"
    CATEGORY = "Anima/Regional/Legacy"

    def apply(
        self,
        model,
        clip,
        regions,
        global_prompt,
        negative_prompt,
        enabled,
        options=None,
    ):
        items = _region_items(regions)
        if not items:
            raise ValueError("[AnimaRegional] no character regions were supplied")

        global_text = str(global_prompt).strip()
        negative_text = str(negative_prompt).strip()
        positive = encode_prompt(clip, global_text, "global")
        negative = encode_prompt(clip, negative_text, "negative")
        if not enabled:
            return (model, positive, negative, "disabled; original model returned")

        diffusion_model = get_diffusion_model(model)
        valid, block_count, reason = validate_anima(diffusion_model)
        if not valid:
            raise ValueError(f"[AnimaRegional] unsupported model: {reason}")

        resolved_regions = []
        for index, item in enumerate(items):
            combined = item["prompt"]
            if global_text:
                combined = f"{combined}\n{global_text}"
            conditioning = encode_prompt(
                clip, combined, f"character {index + 1}"
            )
            resolved = dict(item)
            resolved["conditioning"] = extract_conditioning(
                conditioning, f"character {index + 1}"
            )
            resolved_regions.append(resolved)

        opts = dict(options or {})
        start_block = max(0, int(opts.get("start_block", 0)))
        requested_end = int(opts.get("end_block", -1))
        end_block = block_count - 1 if requested_end < 0 else min(
            block_count - 1, requested_end
        )
        if start_block > end_block:
            raise ValueError(
                f"[AnimaRegional] invalid block range {start_block}-{end_block} "
                f"for a {block_count}-block model"
            )
        target_blocks = list(range(start_block, end_block + 1))

        self_attention_mode = str(opts.get("self_attention_mode", "off"))
        if self_attention_mode == "strict_experimental" and opts.get(
            "overlap_mode", "exclusive"
        ) != "exclusive":
            raise ValueError(
                "[AnimaRegional] strict self-attention requires exclusive overlap"
            )
        self_start = max(0, int(opts.get("self_attn_start_block", 8)))
        requested_self_end = int(opts.get("self_attn_end_block", -1))
        self_end = block_count - 1 if requested_self_end < 0 else min(
            block_count - 1, requested_self_end
        )
        if self_attention_mode == "strict_experimental" and self_start > self_end:
            raise ValueError(
                f"[AnimaRegional] invalid self-attention block range "
                f"{self_start}-{self_end}"
            )
        self_target_blocks = (
            list(range(self_start, self_end + 1))
            if self_attention_mode == "strict_experimental"
            else []
        )

        start_percent = float(opts.get("start_percent", 0.0))
        end_percent = float(opts.get("end_percent", 1.0))
        if start_percent > end_percent:
            raise ValueError("[AnimaRegional] start_percent must not exceed end_percent")
        sigma_range = _resolve_sigma_range(model, start_percent, end_percent)

        existing_patches = getattr(model, "object_patches", None) or {}
        conflicts = [
            path
            for path in existing_patches
            if str(path).endswith(".cross_attn.forward")
            and any(
                str(path).endswith(f"blocks.{index}.cross_attn.forward")
                for index in target_blocks
            )
        ]
        if conflicts:
            raise ValueError(
                "[AnimaRegional] the input model already has cross-attention "
                "patches. Do not chain this node with Anima Artist Mixer or "
                "another regional router."
            )
        self_conflicts = [
            path
            for path in existing_patches
            if str(path).endswith(".self_attn.forward")
            and any(
                str(path).endswith(f"blocks.{index}.self_attn.forward")
                for index in self_target_blocks
            )
        ]
        if self_conflicts:
            raise ValueError(
                "[AnimaRegional] the input model already has self-attention "
                "patches in the requested strict range"
            )

        patched = model.clone()
        state = {
            "enabled": True,
            "regions": resolved_regions,
            "diffusion_model": diffusion_model,
            "overlap_mode": opts.get("overlap_mode", "exclusive"),
            "boundary_falloff": int(opts.get("boundary_falloff", 0)),
            "blend_mode": opts.get("blend_mode", "replace"),
            "self_attention_mode": self_attention_mode,
            "global_strength": float(opts.get("global_strength", 1.0)),
            "branch_chunk_size": int(opts.get("branch_chunk_size", 2)),
            "patch_spatial": int(getattr(diffusion_model, "patch_spatial", 2)),
            "patch_temporal": int(getattr(diffusion_model, "patch_temporal", 1)),
            "sigma_range": sigma_range,
            "current_sigma": None,
            "last_sigma": None,
            "input_shape": None,
            "cond_or_uncond": None,
            "context_cache": {},
            "mask_cache": {},
            "disabled_layers": set(),
            "q_reuse_failed": False,
            "q_reuse_validated": None,
            "q_reuse_logged": False,
            "self_attn_max_tokens": int(opts.get("self_attn_max_tokens", 4096)),
            "self_attn_size_warned": False,
        }

        previous_wrapper = patched.model_options.get("model_function_wrapper")
        if previous_wrapper is not None:
            logger.warning(
                "[AnimaRegional] preserving an existing model wrapper. "
                "Do not chain Artist Mixer while validating regional isolation."
            )
        patched.set_model_unet_function_wrapper(
            make_runtime_capture(state, previous_wrapper)
        )

        for index in target_blocks:
            cross_attn = diffusion_model.blocks[index].cross_attn
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.cross_attn.forward",
                make_cross_attention_patch(
                    cross_attn.forward,
                    cross_attn,
                    state,
                    index,
                ),
            )

        for index in self_target_blocks:
            self_attn = diffusion_model.blocks[index].self_attn
            required = ("compute_qkv", "output_proj", "output_dropout")
            if not all(hasattr(self_attn, name) for name in required):
                raise ValueError(
                    f"[AnimaRegional] block {index} self-attention interface "
                    "is incompatible with strict mode"
                )
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.self_attn.forward",
                make_self_attention_patch(
                    self_attn.forward,
                    self_attn,
                    state,
                    index,
                ),
            )

        status = (
            f"active: {len(resolved_regions)} regions, blocks "
            f"{start_block}-{end_block}, overlap={state['overlap_mode']}, "
            f"blend={state['blend_mode']}, boundary_falloff="
            f"{state['boundary_falloff']}, self_attn={self_attention_mode}"
        )
        return (patched, positive, negative, status)


class AnimaRegionalPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regions": ("ANIMA_REGIONS",),
                "width": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 2048, "step": 8},
                ),
                "overlap_mode": (
                    ["exclusive", "normalized"],
                    {"default": "exclusive"},
                ),
            },
            "optional": {
                "selected_region": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_REGIONS, "step": 1},
                ),
                "boundary_falloff": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "ownership_preview",
        "region_masks",
        "box_masks_image",
        "effective_masks_image",
    )
    FUNCTION = "render"
    CATEGORY = "Anima/Regional/Legacy"

    def render(
        self,
        regions,
        width,
        height,
        overlap_mode,
        selected_region=0,
        boundary_falloff=0,
    ):
        items = _region_items(regions)
        if not items:
            raise ValueError("[AnimaRegional] no regions to preview")
        token_height = max(1, (int(height) + 15) // 16)
        token_width = max(1, (int(width) + 15) // 16)
        token_weights = build_region_weights(
            items,
            1,
            1,
            token_height,
            token_width,
            torch.device("cpu"),
            torch.float32,
            overlap_mode=str(overlap_mode),
            boundary_falloff=int(boundary_falloff),
        )
        raw_weights = build_raw_region_weights(
            items,
            1,
            1,
            int(height),
            int(width),
            torch.device("cpu"),
            torch.float32,
        )
        token_masks = token_weights[:, 0, :, 0].reshape(
            len(items), token_height, token_width
        )
        masks = F.interpolate(
            token_masks.unsqueeze(1),
            size=(int(height), int(width)),
            mode="nearest",
        )[:, 0]
        raw_masks = raw_weights[:, 0, :, 0].reshape(
            len(items), int(height), int(width)
        )
        palette = torch.tensor(
            [
                [0.95, 0.24, 0.22],
                [0.10, 0.65, 0.92],
                [0.20, 0.78, 0.42],
                [0.96, 0.67, 0.13],
                [0.67, 0.35, 0.88],
                [0.93, 0.35, 0.68],
                [0.16, 0.76, 0.72],
                [0.75, 0.80, 0.22],
            ],
            dtype=torch.float32,
        )
        image = torch.full((int(height), int(width), 3), 0.06)
        for index, mask in enumerate(masks):
            image = image + mask.unsqueeze(-1) * palette[index % len(palette)]
        image = image.clamp(0.0, 1.0).unsqueeze(0)
        selected = int(selected_region)
        if selected < 0 or selected > len(items):
            raise ValueError(
                f"[AnimaRegional] selected_region must be 0-{len(items)}, "
                f"got {selected}"
            )
        raw_display = raw_masks
        effective_display = masks
        if selected > 0:
            raw_display = raw_display[selected - 1 : selected]
            effective_display = effective_display[selected - 1 : selected]

        box_masks_image = raw_display.clamp(0.0, 1.0).unsqueeze(-1).expand(
            -1, -1, -1, 3
        )
        effective_masks_image = effective_display.clamp(0.0, 1.0).unsqueeze(
            -1
        ).expand(-1, -1, -1, 3)
        return (image, masks, box_masks_image, effective_masks_image)
