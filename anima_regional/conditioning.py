"""Conditioning extraction and Anima LLMAdapter preprocessing."""

from __future__ import annotations

import torch


def encode_prompt(clip, prompt: str, label: str):
    try:
        tokens = clip.tokenize(prompt)
        return clip.encode_from_tokens_scheduled(tokens)
    except Exception as exc:
        raise ValueError(
            f"[AnimaRegional] failed to encode {label} prompt: {exc}"
        ) from exc


def extract_conditioning(conditioning, label: str):
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError(f"[AnimaRegional] {label} conditioning is empty")
    if len(conditioning) != 1:
        raise ValueError(
            f"[AnimaRegional] {label} produced {len(conditioning)} scheduled "
            "conditioning entries. Prompt scheduling is not supported yet."
        )

    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or not entry:
        raise ValueError(f"[AnimaRegional] {label} conditioning has an invalid shape")
    raw = entry[0]
    if not torch.is_tensor(raw):
        raise ValueError(f"[AnimaRegional] {label} has no tensor embedding")
    extra = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
    return {
        "raw": raw,
        "ids": extra.get("t5xxl_ids"),
        "weights": extra.get("t5xxl_weights"),
    }


def _as_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dim() >= 3 else tensor.unsqueeze(0)


def preprocess_conditioning(diffusion_model, spec, device, dtype):
    raw = _as_batch(spec["raw"]).to(device=device, dtype=dtype)
    ids = spec.get("ids")
    weights = spec.get("weights")

    if ids is None:
        return raw

    ids = ids if ids.dim() >= 2 else ids.unsqueeze(0)
    ids = ids.to(device=device)

    prepared_weights = None
    if weights is not None:
        if weights.dim() == 1:
            prepared_weights = weights.unsqueeze(0).unsqueeze(-1)
        elif weights.dim() == 2:
            prepared_weights = weights.unsqueeze(-1)
        else:
            prepared_weights = weights
        prepared_weights = prepared_weights.to(device=device, dtype=dtype)

    with torch.inference_mode():
        return diffusion_model.preprocess_text_embeds(
            raw,
            ids,
            t5xxl_weights=prepared_weights,
        )


def broadcast_context(context: torch.Tensor, batch_size: int) -> torch.Tensor:
    if context.shape[0] == batch_size:
        return context
    if context.shape[0] == 1:
        return context.expand(batch_size, -1, -1)
    if batch_size % context.shape[0] == 0:
        return context.repeat(batch_size // context.shape[0], 1, 1)
    raise ValueError(
        f"[AnimaRegional] conditioning batch {context.shape[0]} cannot expand "
        f"to model batch {batch_size}"
    )
