"""Runtime model wrappers for spatial cross-attention routing."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .conditioning import broadcast_context, preprocess_conditioning
from .masks import build_region_weights, infer_grid_shape

logger = logging.getLogger(__name__)


def get_diffusion_model(model):
    try:
        return model.get_model_object("diffusion_model")
    except Exception:
        return model.model.diffusion_model


def validate_anima(diffusion_model):
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return False, 0, "diffusion model has no transformer blocks"
    if not callable(getattr(diffusion_model, "preprocess_text_embeds", None)):
        return False, 0, "missing Anima preprocess_text_embeds interface"
    first = blocks[0]
    cross_attn = getattr(first, "cross_attn", None)
    if cross_attn is None:
        return False, 0, "transformer blocks have no cross_attn module"
    required = ("q_proj", "k_proj", "v_proj")
    if not all(hasattr(cross_attn, name) for name in required):
        return False, 0, "cross_attn projection interface is incompatible"
    return True, len(blocks), "ok"


def _expand_cond_flags(cond_or_uncond, batch_size: int):
    if cond_or_uncond is None or len(cond_or_uncond) == 0:
        return [True] * batch_size
    if len(cond_or_uncond) == batch_size:
        return [marker == 0 for marker in cond_or_uncond]
    if batch_size % len(cond_or_uncond) == 0:
        chunk = batch_size // len(cond_or_uncond)
        flags = []
        for marker in cond_or_uncond:
            flags.extend([marker == 0] * chunk)
        return flags
    logger.warning(
        "[AnimaRegional] unusable cond_or_uncond markers %s for batch %d; "
        "treating all rows as conditional",
        cond_or_uncond,
        batch_size,
    )
    return [True] * batch_size


def _call_underlying(previous_wrapper, apply_model, options):
    if previous_wrapper is not None:
        return previous_wrapper(apply_model, options)
    return apply_model(
        options["input"],
        options["timestep"],
        **options["c"],
    )


def make_runtime_capture(state, previous_wrapper):
    def wrapper(apply_model, options):
        input_tensor = options.get("input")
        if torch.is_tensor(input_tensor):
            state["input_shape"] = tuple(int(value) for value in input_tensor.shape)
        state["cond_or_uncond"] = options.get("cond_or_uncond")

        timestep = options.get("timestep")
        if torch.is_tensor(timestep) and timestep.numel() > 0:
            current = float(timestep.flatten()[0].detach().item())
            previous = state.get("last_sigma")
            if previous is not None and current > previous + 1e-3:
                state["disabled_layers"] = set()
            state["current_sigma"] = current
            state["last_sigma"] = current
        return _call_underlying(previous_wrapper, apply_model, options)

    return wrapper


def _sigma_is_active(state):
    sigma_range = state.get("sigma_range")
    current = state.get("current_sigma")
    if sigma_range is None or current is None:
        return True
    return sigma_range[0] <= current <= sigma_range[1]


def _project_perpendicular(delta: torch.Tensor, base: torch.Tensor):
    delta_f = delta.to(torch.float32)
    base_f = base.to(torch.float32)
    denominator = base_f.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    parallel = (delta_f * base_f).sum(dim=-1, keepdim=True) / denominator
    return (delta_f - parallel * base_f).to(delta.dtype)


def _repeat_transformer_options(options, repeat_count: int):
    repeated = dict(options) if isinstance(options, dict) else {}
    markers = repeated.get("cond_or_uncond")
    if markers is not None and repeat_count > 1:
        repeated["cond_or_uncond"] = list(markers) * repeat_count
    return repeated


def _get_region_weights(state, x, reference):
    input_shape = state.get("input_shape")
    patch_spatial = int(state.get("patch_spatial", 2))
    patch_temporal = int(state.get("patch_temporal", 1))
    temporal, grid_h, grid_w = infer_grid_shape(
        int(x.shape[1]),
        input_shape,
        patch_spatial=patch_spatial,
        patch_temporal=patch_temporal,
    )
    cache_key = (
        int(x.shape[0]),
        int(x.shape[1]),
        tuple(input_shape or ()),
        str(reference.device),
        str(reference.dtype),
        state["overlap_mode"],
        int(state.get("boundary_falloff", 0)),
    )
    cache = state.setdefault("mask_cache", {})
    weights = cache.get(cache_key)
    if weights is None:
        weights = build_region_weights(
            state["regions"],
            int(x.shape[0]),
            temporal,
            grid_h,
            grid_w,
            reference.device,
            reference.dtype,
            overlap_mode=state["overlap_mode"],
            boundary_falloff=int(state.get("boundary_falloff", 0)),
        )
        cache[cache_key] = weights
    return weights


def build_self_attention_allow_mask(state, x, reference, cond_or_uncond=None):
    """Build a directed ownership mask with no character-to-character edges."""
    weights = _get_region_weights(state, x, reference).squeeze(-1)
    active = torch.tensor(
        [float(region.get("strength", 1.0)) > 0.0 for region in state["regions"]],
        device=weights.device,
        dtype=torch.bool,
    ).view(-1, 1, 1)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    present = active_weights.max(dim=0).values > 1e-4
    owners = active_weights.argmax(dim=0) + 1
    owners = torch.where(present, owners, torch.zeros_like(owners))

    query_owner = owners.unsqueeze(-1)
    key_owner = owners.unsqueeze(-2)
    same_owner = query_owner == key_owner
    character_reads_background = (query_owner > 0) & (key_owner == 0)
    allowed = same_owner | character_reads_background

    cond_flags = _expand_cond_flags(cond_or_uncond, int(x.shape[0]))
    conditional = torch.tensor(
        cond_flags, device=allowed.device, dtype=torch.bool
    ).view(int(x.shape[0]), 1, 1)
    allowed = torch.where(conditional, allowed, torch.ones_like(allowed))
    return allowed.unsqueeze(1)


class RegionalCrossAttention:
    def __init__(self, original_forward, original_module, state, layer_index: int):
        self.original = original_forward
        self.module = original_module
        self.state = state
        self.layer_index = int(layer_index)

    def _processed_contexts(self, reference_context):
        key = (str(reference_context.device), str(reference_context.dtype))
        cache = self.state.setdefault("context_cache", {})
        cached = cache.get(key)
        if cached is None:
            cached = []
            for region in self.state["regions"]:
                context = preprocess_conditioning(
                    self.state["diffusion_model"],
                    region["conditioning"],
                    reference_context.device,
                    reference_context.dtype,
                ).detach()
                cached.append(context)
            cache[key] = cached
        return [
            broadcast_context(context, int(reference_context.shape[0]))
            for context in cached
        ]

    def _supports_q_reuse(self):
        module = self.module
        return bool(
            module is not None
            and not getattr(module, "is_selfattn", True)
            and all(
                hasattr(module, name)
                for name in (
                    "q_proj",
                    "q_norm",
                    "k_proj",
                    "k_norm",
                    "v_proj",
                    "compute_attention",
                    "n_heads",
                    "head_dim",
                )
            )
        )

    def _q_reuse_chunk(self, x, contexts, transformer_options):
        module = self.module
        count = len(contexts)
        batch_size = int(x.shape[0])
        stacked = torch.cat(contexts, dim=0)
        q_shape = (*x.shape[:-1], module.n_heads, module.head_dim)
        kv_shape = (*stacked.shape[:-1], module.n_heads, module.head_dim)
        q = module.q_norm(module.q_proj(x).view(q_shape))
        k = module.k_norm(module.k_proj(stacked).view(kv_shape))
        v = module.v_proj(stacked).view(kv_shape)
        v_norm = getattr(module, "v_norm", None)
        if v_norm is not None:
            v = v_norm(v)
        q = q.repeat(count, *([1] * (q.dim() - 1)))
        output = module.compute_attention(
            q,
            k,
            v,
            transformer_options=_repeat_transformer_options(
                transformer_options, count
            ),
        )
        return [part for part in output.reshape(count, batch_size, *output.shape[1:])]

    def _branch_outputs(self, x, contexts, rope_emb, transformer_options):
        lengths = {int(context.shape[1]) for context in contexts}
        if (
            self._supports_q_reuse()
            and len(lengths) == 1
            and self.state.get("inner_attention_kinds", {}).get(
                self.layer_index
            ) != "adapter_anchor_q"
            and self.state.get("q_reuse_validated") is not False
            and not self.state.get("q_reuse_failed", False)
        ):
            chunk_size = max(
                1,
                min(int(self.state.get("branch_chunk_size", 2)), len(contexts)),
            )
            try:
                outputs = []
                for start in range(0, len(contexts), chunk_size):
                    outputs.extend(
                        self._q_reuse_chunk(
                            x,
                            contexts[start : start + chunk_size],
                            transformer_options,
                        )
                    )
                validated = self.state.get("q_reuse_validated")
                if validated is None:
                    reference = self.original(
                        x,
                        contexts[0],
                        rope_emb=rope_emb,
                        transformer_options=transformer_options,
                    )
                    tolerance = (
                        2e-3
                        if x.dtype in (torch.float16, torch.bfloat16)
                        else 1e-5
                    )
                    validated = bool(
                        torch.allclose(
                            outputs[0],
                            reference,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                    )
                    self.state["q_reuse_validated"] = validated
                    if not validated:
                        self.state["q_reuse_failed"] = True
                        logger.warning(
                            "[AnimaRegional] Q reuse validation differed from "
                            "the original path; using sequential attention"
                        )
                        return [reference] + [
                            self.original(
                                x,
                                context,
                                rope_emb=rope_emb,
                                transformer_options=transformer_options,
                            )
                            for context in contexts[1:]
                        ]
                if not self.state.get("q_reuse_logged", False):
                    logger.info(
                        "[AnimaRegional] cross-attention Q reuse enabled "
                        "with branch chunk size %d",
                        chunk_size,
                    )
                    self.state["q_reuse_logged"] = True
                return outputs
            except Exception as exc:
                if not self.state.get("q_reuse_failed", False):
                    logger.warning(
                        "[AnimaRegional] Q reuse failed; using sequential "
                        "regional attention: %s",
                        exc,
                    )
                    self.state["q_reuse_failed"] = True

        return [
            self.original(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )
            for context in contexts
        ]

    def _region_weights(self, x, output):
        return _get_region_weights(self.state, x, output)

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        options = transformer_options or {}
        if context is None or not self.state.get("enabled", True):
            return self.original(
                x, context, rope_emb=rope_emb, transformer_options=options
            )
        if self.layer_index in self.state.get("disabled_layers", set()):
            return self.original(
                x, context, rope_emb=rope_emb, transformer_options=options
            )
        if not _sigma_is_active(self.state):
            return self.original(
                x, context, rope_emb=rope_emb, transformer_options=options
            )

        base_output = self.original(
            x,
            context,
            rope_emb=rope_emb,
            transformer_options=options,
        )
        region_contexts = self._processed_contexts(context)
        region_outputs = self._branch_outputs(
            x, region_contexts, rope_emb, options
        )
        weights = self._region_weights(x, base_output)

        markers = options.get("cond_or_uncond")
        if markers is None:
            markers = self.state.get("cond_or_uncond")
        cond_flags = _expand_cond_flags(markers, int(x.shape[0]))
        cond_mask = torch.tensor(
            cond_flags, device=base_output.device, dtype=base_output.dtype
        ).view(1, int(x.shape[0]), 1, 1)
        weights = weights * cond_mask

        routed_delta = torch.zeros_like(base_output)
        blend_mode = self.state.get("blend_mode", "replace")
        global_strength = float(self.state.get("global_strength", 1.0))
        for index, (region, region_output) in enumerate(
            zip(self.state["regions"], region_outputs)
        ):
            delta = region_output - base_output
            if blend_mode == "base_preserve":
                delta = _project_perpendicular(delta, base_output)
            strength = global_strength * float(region.get("strength", 1.0))
            routed_delta = routed_delta + weights[index] * strength * delta
        return base_output + routed_delta


class RegionalForwardPatch:
    _anima_regional_cross_attn_patch = True

    def __init__(self, wrapper: RegionalCrossAttention):
        self.wrapper = wrapper
        self.original_forward = wrapper.original

    def __call__(self, *args, **kwargs):
        return self.wrapper.forward(*args, **kwargs)


def make_cross_attention_patch(original_forward, original_module, state, layer_index):
    return RegionalForwardPatch(
        RegionalCrossAttention(
            original_forward,
            original_module,
            state,
            layer_index,
        )
    )


class StrictRegionalSelfAttention:
    def __init__(self, original_forward, original_module, state, layer_index: int):
        self.original = original_forward
        self.module = original_module
        self.state = state
        self.layer_index = int(layer_index)

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        options = transformer_options or {}
        if self.state.get("self_attention_mode") != "strict_experimental":
            return self.original(
                x, context, rope_emb=rope_emb, transformer_options=options
            )
        if not _sigma_is_active(self.state):
            return self.original(
                x, context, rope_emb=rope_emb, transformer_options=options
            )
        max_tokens = int(self.state.get("self_attn_max_tokens", 4096))
        if int(x.shape[1]) > max_tokens:
            if not self.state.get("self_attn_size_warned", False):
                logger.warning(
                    "[AnimaRegional] strict self-attention skipped at %d tokens; "
                    "limit is %d",
                    int(x.shape[1]),
                    max_tokens,
                )
                self.state["self_attn_size_warned"] = True
            return self.original(
                x, context, rope_emb=rope_emb, transformer_options=options
            )

        module = self.module
        try:
            q, k, v = module.compute_qkv(x, None, rope_emb=rope_emb)
            markers = options.get("cond_or_uncond")
            if markers is None:
                markers = self.state.get("cond_or_uncond")
            allow_mask = build_self_attention_allow_mask(
                self.state, x, x, markers
            )
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allow_mask,
            )
            attended = attended.transpose(1, 2).reshape(
                int(x.shape[0]), int(x.shape[1]), -1
            )
            attended = module.output_proj(attended)
            return module.output_dropout(attended)
        except Exception as exc:
            raise RuntimeError(
                f"[AnimaRegional] strict self-attention failed in block "
                f"{self.layer_index}: {exc}"
            ) from exc


class StrictSelfAttentionForwardPatch:
    _anima_regional_self_attn_patch = True

    def __init__(self, wrapper: StrictRegionalSelfAttention):
        self.wrapper = wrapper
        self.original_forward = wrapper.original

    def __call__(self, *args, **kwargs):
        return self.wrapper.forward(*args, **kwargs)


def make_self_attention_patch(original_forward, original_module, state, layer_index):
    return StrictSelfAttentionForwardPatch(
        StrictRegionalSelfAttention(
            original_forward,
            original_module,
            state,
            layer_index,
        )
    )
