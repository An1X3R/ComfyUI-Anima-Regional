"""Convert normalized boxes and ComfyUI masks to Anima image-token weights."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _factor_spatial_tokens(token_count: int, aspect: float):
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    aspect = max(float(aspect), 1e-6)
    best = (1, token_count)
    best_score = float("inf")
    limit = int(math.sqrt(token_count))
    for height in range(1, limit + 1):
        if token_count % height:
            continue
        width = token_count // height
        for candidate_h, candidate_w in ((height, width), (width, height)):
            score = abs(math.log(max(candidate_w / candidate_h, 1e-6) / aspect))
            if score < best_score:
                best = (candidate_h, candidate_w)
                best_score = score
    return best


def infer_grid_shape(
    sequence_length: int,
    input_shape,
    patch_spatial: int = 2,
    patch_temporal: int = 1,
):
    """Return the T/H/W grid matching Anima's b (t h w) d flatten order."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    shape = tuple(int(value) for value in input_shape) if input_shape else ()
    temporal = 1
    latent_h = latent_w = None
    if len(shape) >= 5:
        temporal = max(1, math.ceil(shape[-3] / max(1, patch_temporal)))
        latent_h, latent_w = shape[-2], shape[-1]
    elif len(shape) >= 4:
        latent_h, latent_w = shape[-2], shape[-1]

    if latent_h and latent_w:
        grid_h = math.ceil(latent_h / max(1, patch_spatial))
        grid_w = math.ceil(latent_w / max(1, patch_spatial))
        if temporal * grid_h * grid_w == sequence_length:
            return temporal, grid_h, grid_w

    if sequence_length % temporal:
        temporal = 1
    spatial_tokens = sequence_length // temporal
    aspect = (latent_w / latent_h) if latent_h and latent_w else 1.0
    grid_h, grid_w = _factor_spatial_tokens(spatial_tokens, aspect)
    if temporal * grid_h * grid_w != sequence_length:
        raise ValueError(
            f"cannot infer a T/H/W grid for {sequence_length} image tokens"
        )
    return temporal, grid_h, grid_w


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def box_mask(region, grid_h: int, grid_w: int, device, dtype):
    center_x = float(region["center_x"])
    center_y = float(region["center_y"])
    width = max(1e-6, float(region["width"]))
    height = max(1e-6, float(region["height"]))
    feather = max(0.0, float(region.get("feather", 0.0)))

    x = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / grid_w
    y = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / grid_h
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    signed_x = width * 0.5 - torch.abs(xx - center_x)
    signed_y = height * 0.5 - torch.abs(yy - center_y)
    signed = torch.minimum(signed_x, signed_y)

    if feather <= 0.0:
        mask = (signed >= 0.0).to(torch.float32)
    else:
        mask = _smoothstep((signed + feather) / (2.0 * feather))
    return mask.to(dtype=dtype)


def supplied_mask(region, batch_size: int, grid_h: int, grid_w: int, device, dtype):
    mask = region.get("mask")
    if not torch.is_tensor(mask):
        raise ValueError("mask region has no mask tensor")
    mask = mask.detach().to(device=device, dtype=torch.float32)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.dim() != 3:
        raise ValueError(f"mask tensor must be HxW or BxHxW, got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    elif mask.shape[0] != batch_size:
        if batch_size % mask.shape[0] == 0:
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1)
        else:
            mask = mask[:1].expand(batch_size, -1, -1)
    mask = F.interpolate(
        mask.unsqueeze(1),
        size=(grid_h, grid_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return mask.clamp(0.0, 1.0).to(dtype=dtype)


def build_raw_region_weights(
    regions,
    batch_size: int,
    temporal: int,
    grid_h: int,
    grid_w: int,
    device,
    dtype,
):
    """Return one un-arbitrated spatial mask per region."""
    raw = []
    for region in regions:
        if region["geometry"] == "mask":
            spatial = supplied_mask(
                region, batch_size, grid_h, grid_w, device, dtype
            )
        else:
            spatial = box_mask(region, grid_h, grid_w, device, dtype)
            spatial = spatial.unsqueeze(0).expand(batch_size, -1, -1)
        flat = spatial.unsqueeze(1).expand(-1, temporal, -1, -1).reshape(
            batch_size, temporal * grid_h * grid_w, 1
        )
        raw.append(flat)

    if not raw:
        raise ValueError("at least one region is required")
    return torch.stack(raw, dim=0)


def build_region_weights(
    regions,
    batch_size: int,
    temporal: int,
    grid_h: int,
    grid_w: int,
    device,
    dtype,
    overlap_mode: str = "exclusive",
    boundary_falloff: int = 0,
):
    stacked = build_raw_region_weights(
        regions,
        batch_size,
        temporal,
        grid_h,
        grid_w,
        device,
        dtype,
    )

    if overlap_mode == "exclusive":
        x = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / grid_w
        y = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / grid_h
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        scores = []
        for index, region in enumerate(regions):
            anchor_x = float(region.get("anchor_x", region.get("center_x", 0.5)))
            anchor_y = float(region.get("anchor_y", region.get("center_y", 0.5)))
            distance = (xx - anchor_x).square() + (yy - anchor_y).square()
            distance = distance.unsqueeze(0).unsqueeze(0).expand(
                batch_size, temporal, -1, -1
            ).reshape(batch_size, temporal * grid_h * grid_w, 1)
            priority = float(region.get("priority", 0))
            score = stacked[index].to(torch.float32) + priority * 4.0
            score = score - distance * 1e-3 - index * 1e-7
            score = torch.where(
                stacked[index] > 1e-4,
                score,
                torch.full_like(score, -torch.inf),
            )
            scores.append(score)
        score_stack = torch.stack(scores, dim=0)
        winner = score_stack.argmax(dim=0, keepdim=True)
        region_ids = torch.arange(
            len(regions), device=device, dtype=winner.dtype
        ).view(-1, 1, 1, 1)
        ownership = region_ids == winner
        any_active = torch.isfinite(score_stack).any(dim=0, keepdim=True)
        ownership_strength = ownership.to(dtype=stacked.dtype)
        radius = max(0, int(boundary_falloff))
        if radius > 0 and len(regions) > 1:
            region_count = len(regions)
            owner_maps = ownership_strength.reshape(
                region_count, batch_size, temporal, grid_h, grid_w
            )
            domain_maps = any_active.to(dtype=stacked.dtype).expand(
                region_count, -1, -1, -1
            ).reshape(region_count, batch_size, temporal, grid_h, grid_w)
            owner_flat = owner_maps.reshape(-1, 1, grid_h, grid_w)
            domain_flat = domain_maps.reshape(-1, 1, grid_h, grid_w)
            kernel = radius * 2 + 1
            owner_average = F.avg_pool2d(
                owner_flat, kernel_size=kernel, stride=1, padding=radius
            )
            domain_average = F.avg_pool2d(
                domain_flat, kernel_size=kernel, stride=1, padding=radius
            )
            purity = (owner_average / domain_average.clamp_min(1e-6)).clamp(
                0.0, 1.0
            )
            ownership_strength = purity.reshape(
                region_count, batch_size, temporal * grid_h * grid_w, 1
            ) * ownership_strength
        return stacked * ownership_strength * any_active

    if overlap_mode != "normalized":
        raise ValueError(f"unknown overlap_mode {overlap_mode!r}")
    total = stacked.sum(dim=0, keepdim=True)
    scale = torch.where(total > 1.0, total.reciprocal(), torch.ones_like(total))
    return stacked * scale
