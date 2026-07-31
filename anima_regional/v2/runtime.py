"""V2 character-level cross-attention router.

This deliberately keeps the V2 ownership model separate from the legacy
region-per-prompt router: each branch is a character, and its body boxes have
already been max-unioned before attention is evaluated.
"""

from __future__ import annotations

import copy
import math

import torch

from ..conditioning import broadcast_context, preprocess_conditioning
from ..masks import infer_grid_shape
from ..runtime import (
    _expand_cond_flags,
    _project_perpendicular,
    _sigma_is_active,
    get_diffusion_model,
    make_runtime_capture,
    validate_anima,
)
from .masks import build_character_masks, build_identity_core_masks
from .validation import _error


EXTENDED_DELTA_NORM_CAP = 4.0
MULTI_CHARACTER_GUARDS = ("off", "soft", "strong")
DETAIL_PRESERVE_MODES = ("off", "soft", "strong")
IDENTITY_ANCHOR_MODES = ("off", "shared_delta")
IDENTITY_DETAIL_MODES = ("off", "late")
IDENTITY_DETAIL_NORM_CAP = 1.5


def _multi_character_profile(state):
    """Return a local-overlap guard, never a global character-count scale."""
    mode = str(state.get("multi_character_guard", "off"))
    if mode == "off":
        return 1.0, None
    if mode == "soft":
        return 1.0, 2.0
    if mode == "strong":
        return 1.0, 1.5
    return 1.0, None


def _composition_profile(state):
    """Return (multiplier, body expansion) for the current diffusion phase."""
    if state.get("composition_mode", "off") != "early_layout":
        return 1.0, 0.0
    sigma = state.get("current_sigma")
    sigma_range = state.get("composition_sigma_range")
    if sigma is None or not sigma_range:
        return 1.0, 0.0
    low, high = (float(sigma_range[0]), float(sigma_range[1]))
    span = max(high - low, 1e-6)
    progress = (high - float(sigma)) / span
    progress = max(0.0, min(1.0, progress))
    cutoff = max(0.05, min(1.0, float(state.get("composition_end_percent", 0.55))))
    if progress >= cutoff:
        return 1.0, 0.0
    remaining = 1.0 - (progress / cutoff)
    remaining = remaining * remaining * (3.0 - 2.0 * remaining)
    strength = max(0.0, min(2.0, float(state.get("composition_strength", 1.0))))
    multiplier = 1.0 + (0.5 * strength * remaining)
    expansion = max(0.0, min(0.15, float(state.get("composition_expand", 0.0)))) * remaining
    return multiplier, expansion


def _sampling_progress(state):
    """Return normalized sampling progress when the model exposes sigma."""
    sigma = state.get("current_sigma")
    sigma_range = state.get("sampling_sigma_range") or state.get(
        "detail_sigma_range"
    )
    if sigma is None or not sigma_range:
        return None
    low, high = (float(sigma_range[0]), float(sigma_range[1]))
    span = max(high - low, 1e-6)
    return max(0.0, min(1.0, (high - float(sigma)) / span))


def _identity_detail_scale(state):
    """Legacy-only late identity schedule.

    ``separated_v1`` does not call this helper: its identity branch is already
    part of the single background-relative residual budget below.  Keep the
    helper for saved legacy packs and tests, but do not use it as a second
    identity injection in the new route.
    """
    if str(state.get("identity_detail_mode", "off")) != "late":
        return 0.0
    progress = _sampling_progress(state)
    if progress is None:
        return 0.0
    start = max(
        0.0,
        min(0.95, float(state.get("identity_detail_start", 0.6))),
    )
    if progress <= start:
        return 0.0
    phase = (progress - start) / max(1.0 - start, 1e-6)
    phase = phase * phase * (3.0 - 2.0 * phase)
    strength = max(
        0.0,
        min(2.0, float(state.get("identity_detail_strength", 0.65))),
    )
    return strength * phase


def _detail_preserve_scale(state):
    """Fade regional residuals late so the base model can finish fine detail."""
    mode = str(state.get("detail_preserve_mode", "off"))
    if mode == "off":
        return 1.0
    progress = _sampling_progress(state)
    if progress is None:
        return 1.0
    start = max(0.0, min(0.95, float(state.get("detail_preserve_start", 0.65))))
    if progress <= start:
        return 1.0
    amount = max(0.0, min(0.95, float(state.get("detail_preserve_amount", 0.5))))
    if mode == "strong":
        amount = max(amount, 0.7)
    phase = (progress - start) / max(1.0 - start, 1e-6)
    phase = phase * phase * (3.0 - 2.0 * phase)
    return 1.0 - (amount * phase)


def _shape_region_weights(weights, power):
    """Reduce soft edge influence without moving or expanding ownership."""
    power = max(1.0, float(power))
    if power <= 1.0:
        return weights
    return weights.clamp(0.0, 1.0).pow(power)


def _scale_extended_delta(delta, base, strength):
    """Scale identity deltas while capping only the new >2x range."""
    scaled = delta * float(strength)
    if float(strength) <= 2.0:
        return scaled
    delta_norm = scaled.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt()
    base_norm = base.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt()
    limit = base_norm.clamp_min(1e-3) * EXTENDED_DELTA_NORM_CAP
    factor = (limit / delta_norm.clamp_min(1e-6)).clamp(max=1.0)
    return (scaled.to(torch.float32) * factor).to(dtype=scaled.dtype)


def _cap_residual_delta(delta, base, cap_factor):
    """Cap a routed residual relative to the base attention output."""
    if cap_factor is None:
        return delta
    factor = max(0.0, float(cap_factor))
    if factor <= 0.0:
        return torch.zeros_like(delta)
    delta_f = delta.to(torch.float32)
    base_f = base.to(torch.float32)
    delta_norm = delta_f.square().sum(dim=-1, keepdim=True).sqrt()
    base_norm = base_f.square().sum(dim=-1, keepdim=True).sqrt()
    if float(base_norm.detach().amax()) < 1e-5:
        return delta
    limit = base_norm.clamp_min(1e-3) * factor
    scale = (limit / delta_norm.clamp_min(1e-6)).clamp(max=1.0)
    return (delta_f * scale).to(dtype=delta.dtype)


def _blend_identity_residual(main_delta, identity_delta, floor, anchor_strength):
    """Legacy-only identity floor helper.

    The separated route applies its floor directly to ``identity_total`` so
    pose and layout residuals cannot be lifted by the identity setting.
    """
    floor = max(0.0, min(1.0, float(floor)))
    anchor_strength = max(0.0, min(1.0, float(anchor_strength)))
    identity_scale = max(1.0, floor)
    if anchor_strength <= 0.0:
        return main_delta, identity_delta
    return main_delta, identity_delta * (1.0 + (identity_scale - 1.0) * anchor_strength)


def _sigma_range(model, start_percent, end_percent):
    if start_percent == 0.0 and end_percent == 1.0:
        return None
    try:
        sampling = model.get_model_object("model_sampling")
        values = (float(sampling.percent_to_sigma(start_percent)), float(sampling.percent_to_sigma(end_percent)))
    except Exception as exc:
        raise _error(f"failed to resolve sampling range: {exc}") from exc
    return tuple(sorted(values))


def _v2_weights(state, x, reference, body_expand=0.0):
    temporal, grid_h, grid_w = infer_grid_shape(
        int(x.shape[1]), state.get("input_shape"),
        patch_spatial=int(state.get("patch_spatial", 2)),
        patch_temporal=int(state.get("patch_temporal", 1)),
    )
    cache_key = (
        int(x.shape[0]),
        int(x.shape[1]),
        tuple(state.get("input_shape") or ()),
        str(reference.device),
        str(reference.dtype),
        int(state.get("boundary_falloff", 0)),
        round(float(body_expand), 4),
    )
    cache = state.setdefault("mask_cache", {})
    weights = cache.get(cache_key)
    if weights is None:
        raw, ownership = build_character_masks(
            state["layout"],
            grid_h,
            grid_w,
            reference.device,
            reference.dtype,
            boundary_falloff=int(state.get("boundary_falloff", 0)),
            body_expand=float(body_expand),
        )
        weights = ownership.unsqueeze(1).unsqueeze(1).expand(
            -1, int(x.shape[0]), temporal, -1, -1
        ).reshape(len(ownership), int(x.shape[0]), temporal * grid_h * grid_w, 1)
        cache[cache_key] = weights
        overlap = (raw.to(torch.float32).sum(dim=0) - 1.0).clamp(0.0, 1.0)
        overlap = overlap.reshape(1, 1, grid_h * grid_w, 1).expand(
            int(x.shape[0]), temporal, grid_h * grid_w, 1
        ).reshape(1, int(x.shape[0]), temporal * grid_h * grid_w, 1)
        state.setdefault("mask_overlap_cache", {})[cache_key] = overlap.to(
            device=reference.device, dtype=reference.dtype
        )
    state["last_overlap_mask"] = state.setdefault("mask_overlap_cache", {}).get(
        cache_key, torch.zeros((1, int(x.shape[0]), temporal * grid_h * grid_w, 1), device=reference.device, dtype=reference.dtype)
    )
    return weights


def _v2_identity_core_weights(state, x, reference):
    """Legacy-only body-core weights retained for old option payloads.

    The current ``separated_v1`` path intentionally uses the ordinary Body
    ownership mask for identity.  A second core multiplier would silently
    create another strength control and make contact regions harder to tune.
    """
    temporal, grid_h, grid_w = infer_grid_shape(
        int(x.shape[1]),
        state.get("input_shape"),
        patch_spatial=int(state.get("patch_spatial", 2)),
        patch_temporal=int(state.get("patch_temporal", 1)),
    )
    radius = max(
        0.05,
        min(1.0, float(state.get("identity_core_radius", 0.55))),
    )
    cache_key = (
        int(x.shape[0]),
        int(x.shape[1]),
        tuple(state.get("input_shape") or ()),
        str(reference.device),
        str(reference.dtype),
        round(radius, 4),
    )
    cache = state.setdefault("identity_core_cache", {})
    weights = cache.get(cache_key)
    if weights is None:
        core = build_identity_core_masks(
            state["layout"],
            grid_h,
            grid_w,
            radius,
            reference.device,
            reference.dtype,
        )
        weights = core.unsqueeze(1).unsqueeze(1).expand(
            -1,
            int(x.shape[0]),
            temporal,
            -1,
            -1,
        ).reshape(
            len(core),
            int(x.shape[0]),
            temporal * grid_h * grid_w,
            1,
        )
        cache[cache_key] = weights
    return weights


class V2RegionalCrossAttention:
    """Outer regional router; its inner operation may be a Q-only Anchor."""

    def __init__(self, inner_forward, state, layer_index):
        self.inner_forward = inner_forward
        self.state = state
        self.layer_index = int(layer_index)

    def _contexts(self, reference_context):
        key = (str(reference_context.device), str(reference_context.dtype))
        cached = self.state.setdefault("context_cache", {}).get(key)
        if cached is None:
            cached = []
            for character in self.state["characters"]:
                pose_spec = character.get("pose_conditioning")
                identity_spec = character.get("identity_conditioning")
                if (
                    self.state.get("routing_contract") == "separated_v1"
                    and pose_spec is None
                    and isinstance(identity_spec, dict)
                ):
                    cached.append(None)
                    continue
                spec = pose_spec or character.get("conditioning")
                cached.append(preprocess_conditioning(
                    self.state["diffusion_model"], spec,
                    reference_context.device, reference_context.dtype,
                ).detach())
            self.state["context_cache"][key] = cached
        return [
            None
            if value is None
            else broadcast_context(value, int(reference_context.shape[0]))
            for value in cached
        ]

    def _identity_contexts(self, reference_context):
        """Return optional stable-identity contexts, preserving character order."""
        key = (str(reference_context.device), str(reference_context.dtype))
        cached = self.state.setdefault("identity_context_cache", {}).get(key)
        if cached is None:
            cached = []
            for character in self.state["characters"]:
                spec = character.get("identity_conditioning")
                if not isinstance(spec, dict):
                    cached.append(None)
                    continue
                cached.append(
                    preprocess_conditioning(
                        self.state["diffusion_model"],
                        spec,
                        reference_context.device,
                        reference_context.dtype,
                    ).detach()
                )
            self.state["identity_context_cache"][key] = cached
        return [
            None
            if value is None
            else broadcast_context(value, int(reference_context.shape[0]))
            for value in cached
        ]

    def _shared_context(self, reference_context):
        spec = self.state.get("background_conditioning") or self.state.get("shared_conditioning")
        if not isinstance(spec, dict):
            return None
        key = (str(reference_context.device), str(reference_context.dtype))
        cached = self.state.setdefault("shared_context_cache", {}).get(key)
        if cached is None:
            cached = preprocess_conditioning(
                self.state["diffusion_model"],
                spec,
                reference_context.device,
                reference_context.dtype,
            ).detach()
            self.state["shared_context_cache"][key] = cached
        return broadcast_context(cached, int(reference_context.shape[0]))

    def _layout_context(self, reference_context):
        spec = self.state.get("layout_conditioning")
        if not isinstance(spec, dict):
            return None
        key = (str(reference_context.device), str(reference_context.dtype))
        cached = self.state.setdefault("layout_context_cache", {}).get(key)
        if cached is None:
            cached = preprocess_conditioning(
                self.state["diffusion_model"],
                spec,
                reference_context.device,
                reference_context.dtype,
            ).detach()
            self.state["layout_context_cache"][key] = cached
        return broadcast_context(cached, int(reference_context.shape[0]))

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        options = transformer_options or {}
        if (
            context is None
            or self.layer_index in self.state.get("disabled_layers", set())
            or not _sigma_is_active(self.state)
        ):
            return self.inner_forward(x, context, rope_emb=rope_emb, transformer_options=options)

        base = self.inner_forward(x, context, rope_emb=rope_emb, transformer_options=options)
        routing_contract = self.state.get("routing_contract", "legacy_v2")
        contexts = self._contexts(context)
        identity_contexts = (
            self._identity_contexts(context)
            if self.state.get("has_identity_conditioning", False)
            else [None] * len(self.state["characters"])
        )
        identity_anchor_mode = str(self.state.get("identity_anchor_mode", "off"))
        anchor_strength = (
            max(0.0, min(1.0, float(self.state.get("identity_anchor_strength", 1.0))))
            if identity_anchor_mode == "shared_delta"
            else 0.0
        )

        # The actual positive input is the base branch. When it came from an
        # external Mixer, evaluate the standalone background once so character
        # branches can remove that common response without duplicating it.
        background_output = base
        if not self.state.get("background_is_final", self.state.get("shared_is_final", False)):
            background_context = self._shared_context(context)
            if background_context is not None:
                background_output = self.inner_forward(
                    x, background_context, rope_emb=rope_emb, transformer_options=options
                )

        layout_context = self._layout_context(context)
        layout_output = (
            self.inner_forward(x, layout_context, rope_emb=rope_emb, transformer_options=options)
            if layout_context is not None
            else background_output
        )
        composition_multiplier, body_expand = _composition_profile(self.state)
        branch_scale, residual_cap = _multi_character_profile(self.state)
        global_strength = float(self.state.get("global_strength", 1.0))
        has_layout = layout_context is not None
        layout_weights = _v2_weights(self.state, x, base, body_expand=body_expand)
        layout_overlap = self.state.get("last_overlap_mask")
        identity_weights = _v2_weights(self.state, x, base, body_expand=0.0)
        layout_weights = _shape_region_weights(
            layout_weights, self.state.get("edge_focus_power", 1.0)
        )
        identity_weights = _shape_region_weights(
            identity_weights, self.state.get("edge_focus_power", 1.0)
        )
        flags = _expand_cond_flags(
            options.get("cond_or_uncond", self.state.get("cond_or_uncond")),
            int(x.shape[0]),
        )
        conditional = torch.tensor(flags, device=base.device, dtype=base.dtype).view(
            1, int(x.shape[0]), 1, 1
        )
        layout_weights = layout_weights * conditional
        identity_weights = identity_weights * conditional
        layout_union = layout_weights.sum(dim=0).clamp(0.0, 1.0)
        layout_delta = layout_output - background_output
        layout_total = torch.zeros_like(base)
        if has_layout:
            layout_total = layout_union * _scale_extended_delta(
                layout_delta,
                base,
                global_strength
                * max(0.0, float(self.state.get("layout_strength", 1.0)))
                * composition_multiplier,
            )
        pose_total = torch.zeros_like(base)
        identity_total = torch.zeros_like(base)
        legacy_total = torch.zeros_like(base)

        for index, (character, character_context) in enumerate(
            zip(self.state["characters"], contexts)
        ):
            character_strength = float(character["strength"])
            if global_strength == 0.0 or character_strength == 0.0:
                continue

            identity_context = identity_contexts[index]
            if routing_contract == "separated_v1":
                if character_context is not None:
                    pose_branch = self.inner_forward(
                        x, character_context, rope_emb=rope_emb, transformer_options=options
                    )
                    pose_reference = layout_output if has_layout else background_output
                    pose_delta = pose_branch - pose_reference
                    if self.state["blend_mode"] == "base_preserve":
                        pose_delta = _project_perpendicular(pose_delta, base)
                    pose_total = pose_total + layout_weights[index] * _scale_extended_delta(
                        pose_delta,
                        base,
                        global_strength * character_strength * composition_multiplier,
                    )
                if identity_context is not None:
                    identity_branch = self.inner_forward(
                        x, identity_context, rope_emb=rope_emb, transformer_options=options
                    )
                    identity_delta = identity_branch - background_output
                    if self.state["blend_mode"] == "base_preserve":
                        identity_delta = _project_perpendicular(identity_delta, base)
                    identity_total = identity_total + identity_weights[index] * _scale_extended_delta(
                        identity_delta,
                        base,
                        global_strength * character_strength,
                    )
                continue

            if character_context is None:
                continue
            branch = self.inner_forward(
                x, character_context, rope_emb=rope_emb, transformer_options=options
            )
            base_delta = branch - base
            delta = base_delta
            if identity_anchor_mode == "shared_delta" and anchor_strength > 0.0:
                delta = (
                    base_delta * (1.0 - anchor_strength)
                    + (branch - background_output) * anchor_strength
                )
            if self.state["blend_mode"] == "base_preserve":
                delta = _project_perpendicular(delta, base)
            legacy_total = legacy_total + layout_weights[index] * _scale_extended_delta(
                delta,
                base,
                global_strength * character_strength * composition_multiplier * branch_scale,
            )

        detail_scale = _detail_preserve_scale(self.state)
        if routing_contract == "separated_v1":
            # Identity floor applies only to the isolated identity residual.
            identity_scale = max(
                detail_scale,
                max(0.0, min(1.0, float(self.state.get("identity_late_floor", 0.0)))),
            )
            routed = layout_total * detail_scale + pose_total * detail_scale + identity_total * identity_scale
        else:
            routed = legacy_total * detail_scale
        if residual_cap is not None and layout_overlap is not None:
            capped = _cap_residual_delta(routed, base, residual_cap)
            routed = routed * (1.0 - layout_overlap) + capped * layout_overlap
        return base + routed


class V2RegionalForwardPatch:
    _anima_regional_cross_attn_patch = True

    def __init__(self, router):
        self.router = router
        self.original_forward = router.inner_forward

    def __call__(self, *args, **kwargs):
        return self.router.forward(*args, **kwargs)


def apply_v2_regional(model, regional_pack, blend_mode, advanced_options=None):
    """Patch all requested blocks while composing only the approved Anchor-Q API."""
    try:
        from ..compat import resolve_regional_inner_forward
    except ImportError as exc:
        raise _error("routing compatibility helper is unavailable") from exc
    diffusion_model = get_diffusion_model(model)
    valid, block_count, reason = validate_anima(diffusion_model)
    if not valid:
        raise _error(f"unsupported model: {reason}")
    options = dict(advanced_options or {})
    routing_contract = regional_pack.get("routing_contract", "legacy_v2")
    if routing_contract not in ("legacy_v2", "separated_v1"):
        raise _error("regional pack has an unsupported routing contract")
    start_block = max(0, int(options.get("start_block", 0)))
    requested_end = int(options.get("end_block", -1))
    end_block = block_count - 1 if requested_end < 0 else min(block_count - 1, requested_end)
    if start_block > end_block:
        raise _error(f"invalid block range {start_block}-{end_block} for a {block_count}-block model")
    start_percent = float(options.get("start_percent", 0.0))
    end_percent = float(options.get("end_percent", 1.0))
    if not 0.0 <= start_percent <= end_percent <= 1.0:
        raise _error("start_percent and end_percent must be ordered values between 0 and 1")
    global_strength = float(options.get("global_strength", 1.0))
    if not 0.0 <= global_strength <= 2.0:
        raise _error("global_strength must be between 0 and 2")
    boundary_falloff = max(0, int(options.get("boundary_falloff", 0)))
    composition_mode = str(options.get("composition_mode", "off"))
    if composition_mode not in ("off", "early_layout"):
        raise _error("composition_mode must be off or early_layout")
    composition_strength = float(options.get("composition_strength", 1.0))
    if not 0.0 <= composition_strength <= 2.0:
        raise _error("composition_strength must be between 0 and 2")
    composition_expand = float(options.get("composition_expand", 0.04))
    if not 0.0 <= composition_expand <= 0.15:
        raise _error("composition_expand must be between 0 and 0.15")
    composition_end_percent = float(options.get("composition_end_percent", 0.55))
    if not 0.05 <= composition_end_percent <= 1.0:
        raise _error("composition_end_percent must be between 0.05 and 1")
    layout_strength = float(options.get("layout_strength", 1.0))
    if not 0.0 <= layout_strength <= 2.0:
        raise _error("layout_strength must be between 0 and 2")
    multi_character_guard = str(options.get("multi_character_guard", "off"))
    if multi_character_guard not in MULTI_CHARACTER_GUARDS:
        raise _error(
            "multi_character_guard must be off, soft, or strong"
        )
    detail_preserve_mode = str(options.get("detail_preserve_mode", "off"))
    if detail_preserve_mode not in DETAIL_PRESERVE_MODES:
        raise _error("detail_preserve_mode must be off, soft, or strong")
    detail_preserve_start = float(options.get("detail_preserve_start", 0.65))
    if not 0.0 <= detail_preserve_start <= 0.95:
        raise _error("detail_preserve_start must be between 0 and 0.95")
    detail_preserve_amount = float(options.get("detail_preserve_amount", 0.5))
    if not 0.0 <= detail_preserve_amount <= 0.95:
        raise _error("detail_preserve_amount must be between 0 and 0.95")
    edge_focus_power = float(options.get("edge_focus_power", 1.0))
    if not 1.0 <= edge_focus_power <= 4.0:
        raise _error("edge_focus_power must be between 1 and 4")
    identity_anchor_mode = str(options.get("identity_anchor_mode", "off"))
    if identity_anchor_mode not in IDENTITY_ANCHOR_MODES:
        raise _error("identity_anchor_mode must be off or shared_delta")
    identity_anchor_strength = float(options.get("identity_anchor_strength", 1.0))
    if not 0.0 <= identity_anchor_strength <= 1.0:
        raise _error("identity_anchor_strength must be between 0 and 1")
    identity_late_floor = float(options.get("identity_late_floor", 0.8))
    if not 0.0 <= identity_late_floor <= 1.0:
        raise _error("identity_late_floor must be between 0 and 1")
    identity_detail_mode = str(options.get("identity_detail_mode", "off"))
    if identity_detail_mode not in IDENTITY_DETAIL_MODES:
        raise _error("identity_detail_mode must be off or late")
    identity_detail_start = float(options.get("identity_detail_start", 0.6))
    if not 0.0 <= identity_detail_start <= 0.95:
        raise _error("identity_detail_start must be between 0 and 0.95")
    identity_detail_strength = float(
        options.get("identity_detail_strength", 0.65)
    )
    if not 0.0 <= identity_detail_strength <= 2.0:
        raise _error("identity_detail_strength must be between 0 and 2")
    identity_core_strength = float(options.get("identity_core_strength", 1.5))
    if not 1.0 <= identity_core_strength <= 3.0:
        raise _error("identity_core_strength must be between 1 and 3")
    identity_core_radius = float(options.get("identity_core_radius", 0.55))
    if not 0.05 <= identity_core_radius <= 1.0:
        raise _error("identity_core_radius must be between 0.05 and 1")
    if (
        identity_anchor_mode == "shared_delta"
        and not isinstance(
            regional_pack.get("background_conditioning")
            or regional_pack.get("shared_conditioning"),
            dict,
        )
    ):
        raise _error(
            "shared_delta identity anchoring requires a Prompt Pack with "
            "shared scene conditioning"
        )
    sampling_sigma_range = None
    if detail_preserve_mode != "off" or identity_detail_mode == "late":
        try:
            sampling = model.get_model_object("model_sampling")
            sigma_values = (
                float(sampling.percent_to_sigma(0.0)),
                float(sampling.percent_to_sigma(1.0)),
            )
            sampling_sigma_range = tuple(sorted(sigma_values))
        except Exception:
            # Lightweight wrappers without sampling metadata remain unchanged.
            sampling_sigma_range = None
    detail_sigma_range = (
        sampling_sigma_range if detail_preserve_mode != "off" else None
    )
    composition_sigma_range = None
    if composition_mode == "early_layout":
        try:
            sampling = model.get_model_object("model_sampling")
            sigma_values = (
                float(sampling.percent_to_sigma(0.0)),
                float(sampling.percent_to_sigma(1.0)),
            )
            composition_sigma_range = tuple(sorted(sigma_values))
        except Exception:
            # Wrappers without model_sampling remain on the baseline route.
            composition_sigma_range = None
    layout = copy.deepcopy(regional_pack["layout"])
    active_character_ids = {
        region["character_uuid"]
        for region in layout["regions"]
        if region["enabled"]
        and (
            region["type"] == "body_region"
            or layout["overlap_mode"] == "exclusive"
        )
    }
    layout["characters"] = [
        character
        for character in layout["characters"]
        if character["uuid"] in active_character_ids
    ]
    layout["regions"] = [
        region
        for region in layout["regions"]
        if region["character_uuid"] in active_character_ids
    ]
    characters = [
        character
        for character in regional_pack["characters"]
        if character["uuid"] in active_character_ids
    ]
    if not characters:
        raise _error("regional pack has no active character branches")
    has_identity_conditioning = any(
        isinstance(character.get("identity_conditioning"), dict)
        for character in characters
    )
    if (
        identity_detail_mode == "late"
        and identity_detail_strength > 0.0
        and has_identity_conditioning
        and not isinstance(regional_pack.get("shared_conditioning"), dict)
    ):
        raise _error(
            "late identity detail requires a Prompt Pack with shared scene "
            "conditioning"
        )

    patched = model.clone()
    existing_patches = getattr(model, "object_patches", None) or {}
    state = {
        "layout": layout,
        "characters": characters,
        "diffusion_model": diffusion_model,
        "blend_mode": blend_mode,
        "global_strength": global_strength,
        "boundary_falloff": boundary_falloff,
        "composition_mode": composition_mode,
        "composition_strength": composition_strength,
        "composition_expand": composition_expand,
        "composition_end_percent": composition_end_percent,
        "layout_strength": layout_strength,
        "composition_sigma_range": composition_sigma_range,
        "multi_character_guard": multi_character_guard,
        "detail_preserve_mode": detail_preserve_mode,
        "detail_preserve_start": detail_preserve_start,
        "detail_preserve_amount": detail_preserve_amount,
        "detail_sigma_range": detail_sigma_range,
        "sampling_sigma_range": sampling_sigma_range,
        "edge_focus_power": edge_focus_power,
        "routing_contract": routing_contract,
        "identity_anchor_mode": identity_anchor_mode,
        "identity_anchor_strength": identity_anchor_strength,
        "identity_late_floor": identity_late_floor,
        "identity_detail_mode": identity_detail_mode,
        "identity_detail_start": identity_detail_start,
        "identity_detail_strength": identity_detail_strength,
        "identity_core_strength": identity_core_strength,
        "identity_core_radius": identity_core_radius,
        "has_identity_conditioning": has_identity_conditioning,
        "background_conditioning": regional_pack.get("background_conditioning")
        or regional_pack.get("shared_conditioning"),
        "background_is_final": bool(
            regional_pack.get("background_is_final", regional_pack.get("shared_is_final", False))
        ),
        # Legacy names remain in state for third-party inspectors.
        "shared_conditioning": regional_pack.get("shared_conditioning")
        or regional_pack.get("background_conditioning"),
        "shared_is_final": bool(regional_pack.get("shared_is_final", False)),
        "layout_conditioning": regional_pack.get("layout_conditioning"),
        "patch_spatial": int(getattr(diffusion_model, "patch_spatial", 2)),
        "patch_temporal": int(getattr(diffusion_model, "patch_temporal", 1)),
        "sigma_range": _sigma_range(model, start_percent, end_percent),
        "current_sigma": None,
        "last_sigma": None,
        "input_shape": None,
        "cond_or_uncond": None,
        "context_cache": {},
        "identity_context_cache": {},
        "shared_context_cache": {},
        "layout_context_cache": {},
        "mask_cache": {},
        "identity_core_cache": {},
        "disabled_layers": set(),
    }
    previous_wrapper = patched.model_options.get("model_function_wrapper")
    patched.set_model_unet_function_wrapper(make_runtime_capture(state, previous_wrapper))
    for index in range(start_block, end_block + 1):
        path = f"diffusion_model.blocks.{index}.cross_attn.forward"
        inner, kind = resolve_regional_inner_forward(existing_patches, path, diffusion_model.blocks[index].cross_attn.forward)
        router = V2RegionalCrossAttention(inner, state, index)
        patch = V2RegionalForwardPatch(router)
        patch._anima_regional_inner_attention_kind = kind
        patched.add_object_patch(path, patch)
    return patched
