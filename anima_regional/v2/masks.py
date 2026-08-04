"""V2 character-level mask construction and inspection rendering."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _box(region, height: int, width: int, device, dtype):
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    left, top = region["x"], region["y"]
    right, bottom = left + region["width"], top + region["height"]
    inside = (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
    feather = region["feather"]
    if feather <= 0.0:
        return inside.to(dtype)
    distance = torch.minimum(torch.minimum(xx - left, right - xx), torch.minimum(yy - top, bottom - yy))
    ramp = (distance / feather).clamp(0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    return (ramp * inside).to(dtype)


def _expanded_body_region(region, amount: float):
    """Return a temporary, clamped expansion used only during early layout."""
    amount = max(0.0, float(amount))
    if amount <= 0.0:
        return region
    left = max(0.0, float(region["x"]) - amount)
    top = max(0.0, float(region["y"]) - amount)
    right = min(1.0, float(region["x"]) + float(region["width"]) + amount)
    bottom = min(1.0, float(region["y"]) + float(region["height"]) + amount)
    expanded = dict(region)
    expanded.update({
        "x": left,
        "y": top,
        "width": max(0.001, right - left),
        "height": max(0.001, bottom - top),
    })
    return expanded


def _distance_to_region_center(region, yy, xx):
    """Return squared canvas-space distance to a box's original center."""
    center_x = float(region["x"]) + float(region["width"]) * 0.5
    center_y = float(region["y"]) + float(region["height"]) * 0.5
    return (xx - center_x).square() + (yy - center_y).square()


def _stable_uuid_tie_break(ordered, device):
    """Return an order-independent epsilon rank for exact Voronoi ties."""
    rank_by_uuid = {
        identifier: rank
        for rank, identifier in enumerate(sorted(ordered))
    }
    return torch.tensor(
        [rank_by_uuid[identifier] for identifier in ordered],
        device=device,
        dtype=torch.float32,
    ).view(-1, 1, 1) * 1e-7


def build_raw_region_masks(layout, height: int, width: int, device=None, dtype=torch.float32):
    """Return one raw inspection mask for every enabled body or Hint box."""
    device = device or torch.device("cpu")
    masks = [
        _box(region, height, width, device, dtype)
        for region in layout["regions"]
        if region["enabled"]
    ]
    if not masks:
        return torch.zeros((1, height, width), device=device, dtype=dtype)
    return torch.stack(masks)


def _apply_internal_boundary_falloff(effective, owners, active, radius: int):
    radius = max(0, int(radius))
    if radius <= 0 or int(effective.shape[0]) <= 1:
        return effective
    count, height, width = effective.shape
    owner_maps = owners.to(torch.float32).reshape(count, 1, height, width)
    domain_maps = active.to(torch.float32).expand(count, -1, -1).reshape(
        count, 1, height, width
    )
    kernel = radius * 2 + 1
    owner_average = F.avg_pool2d(
        owner_maps, kernel_size=kernel, stride=1, padding=radius
    )
    domain_average = F.avg_pool2d(
        domain_maps, kernel_size=kernel, stride=1, padding=radius
    )
    purity = (owner_average / domain_average.clamp_min(1e-6)).clamp(0.0, 1.0)
    return effective * purity[:, 0].to(device=effective.device, dtype=effective.dtype)


def build_character_mask_components(
    layout,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    boundary_falloff: int = 0,
    body_expand: float = 0.0,
):
    """Return raw Body, final ownership, and Hint-target masks by UUID.

    Equal-priority overlaps retain the automatic nearest-center/Voronoi
    behavior. A higher ``priority`` restores the early front-to-back layer
    override only where regions overlap. Hard Ownership Hints retain the
    historical 0.2 behavior. A Hint with
    ``hint_blend=soft`` crossfades from the current Body/Hard-Hint owner to the
    Hint owner. Outside existing ownership coverage, partial Soft-Hint weight
    intentionally leaves the remainder on the runtime base branch.
    """
    device = device or torch.device("cpu")
    characters = layout["characters"]
    raw = {item["uuid"]: torch.zeros((height, width), device=device, dtype=dtype) for item in characters}
    hard_hints = {
        item["uuid"]: torch.zeros((height, width), device=device, dtype=dtype)
        for item in characters
    }
    soft_hints = {
        item["uuid"]: torch.zeros((height, width), device=device, dtype=dtype)
        for item in characters
    }
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    body_distance = {
        item["uuid"]: torch.full((height, width), torch.inf, device=device)
        for item in characters
    }
    hard_hint_distance = {
        item["uuid"]: torch.full((height, width), torch.inf, device=device)
        for item in characters
    }
    soft_hint_distance = {
        item["uuid"]: torch.full((height, width), torch.inf, device=device)
        for item in characters
    }
    body_priority = {
        item["uuid"]: torch.full((height, width), -torch.inf, device=device)
        for item in characters
    }
    hard_hint_priority = {
        item["uuid"]: torch.full((height, width), -torch.inf, device=device)
        for item in characters
    }
    soft_hint_priority = {
        item["uuid"]: torch.full((height, width), -torch.inf, device=device)
        for item in characters
    }
    for region in layout["regions"]:
        if not region["enabled"]:
            continue
        source = (
            _expanded_body_region(region, body_expand)
            if region["type"] == "body_region"
            else region
        )
        mask = _box(source, height, width, device, dtype)
        is_body = region["type"] == "body_region"
        key = region["character_uuid"]
        priority = float(region.get("priority", 0))
        distance = _distance_to_region_center(region, yy, xx)
        if is_body:
            raw[key] = torch.maximum(raw[key], mask)
            active = mask > 1e-4
            current_priority = body_priority[key]
            higher = active & (priority > current_priority)
            equal = active & (priority == current_priority)
            body_distance[key] = torch.where(
                higher,
                distance,
                torch.where(
                    equal,
                    torch.minimum(body_distance[key], distance),
                    body_distance[key],
                ),
            )
            body_priority[key] = torch.where(
                active,
                torch.maximum(
                    current_priority,
                    torch.full_like(current_priority, priority),
                ),
                current_priority,
            )
            continue

        blend = str(region.get("hint_blend", "hard") or "hard")
        if blend == "soft":
            strength = max(0.0, min(1.0, float(region.get("strength", 1.0))))
            target = soft_hints
            distance_map = soft_hint_distance
            priority_map = soft_hint_priority
            mask = mask * strength
        else:
            target = hard_hints
            distance_map = hard_hint_distance
            priority_map = hard_hint_priority
        target[key] = torch.maximum(target[key], mask)
        active = mask > 1e-4
        current_priority = priority_map[key]
        higher = active & (priority > current_priority)
        equal = active & (priority == current_priority)
        distance_map[key] = torch.where(
            higher,
            distance,
            torch.where(
                equal,
                torch.minimum(distance_map[key], distance),
                distance_map[key],
            ),
        )
        priority_map[key] = torch.where(
            active,
            torch.maximum(
                current_priority,
                torch.full_like(current_priority, priority),
            ),
            current_priority,
        )

    ordered = [item["uuid"] for item in characters]
    raw_stack = torch.stack([raw[key] for key in ordered])
    if layout["overlap_mode"] == "normalized":
        total = raw_stack.sum(dim=0, keepdim=True)
        effective = raw_stack * torch.where(
            total > 1.0,
            total.reciprocal(),
            torch.ones_like(total),
        )
        return raw_stack, effective, torch.zeros_like(effective)

    hard_hint_stack = torch.stack([hard_hints[key] for key in ordered])
    soft_hint_stack = torch.stack([soft_hints[key] for key in ordered])
    body_distance_stack = torch.stack([body_distance[key] for key in ordered])
    hard_hint_distance_stack = torch.stack(
        [hard_hint_distance[key] for key in ordered]
    )
    soft_hint_distance_stack = torch.stack(
        [soft_hint_distance[key] for key in ordered]
    )
    body_priority_stack = torch.stack([body_priority[key] for key in ordered])
    hard_hint_priority_stack = torch.stack(
        [hard_hint_priority[key] for key in ordered]
    )
    soft_hint_priority_stack = torch.stack(
        [soft_hint_priority[key] for key in ordered]
    )
    tie_break = _stable_uuid_tie_break(ordered, device)

    # Priority restores an explicit front-to-back layer only in intersections.
    # Equal priorities keep the nearest-center/Voronoi decision and stable UUID
    # tie-break, so existing layouts remain unchanged at the default value 0.
    body_scores = body_distance_stack - body_priority_stack * 4.0 + tie_break
    body_active = torch.isfinite(body_scores).any(dim=0)
    body_winner = body_scores.argmin(dim=0)

    # Hard Ownership Hints are explicit local overrides.  When Hints
    # themselves intersect, their nearest original center decides without
    # socket order.
    hard_hint_scores = (
        hard_hint_distance_stack - hard_hint_priority_stack * 4.0 + tie_break
    )
    hard_hint_active = torch.isfinite(hard_hint_scores).any(dim=0)
    hard_hint_winner = hard_hint_scores.argmin(dim=0)

    winner = torch.where(hard_hint_active, hard_hint_winner, body_winner)
    active = hard_hint_active | body_active
    owners = torch.arange(len(ordered), device=device).view(-1, 1, 1) == winner.unsqueeze(0)
    coverage = torch.maximum(raw_stack, hard_hint_stack)
    effective = coverage * owners.to(dtype) * active.unsqueeze(0).to(dtype)
    effective = _apply_internal_boundary_falloff(
        effective,
        owners,
        active.unsqueeze(0),
        boundary_falloff,
    )
    hard_hint_targets = effective * hard_hint_active.unsqueeze(0).to(dtype)

    # Soft Hints interpolate from the already resolved Body/Hard-Hint output
    # to the selected complete character branch. Outside existing coverage,
    # their partial weight leaves the runtime remainder on base. Hard Hints
    # win where the two kinds overlap, preserving the historical contract.
    soft_hint_scores = (
        soft_hint_distance_stack - soft_hint_priority_stack * 4.0 + tie_break
    )
    soft_hint_active = torch.isfinite(soft_hint_scores).any(dim=0)
    soft_hint_winner = soft_hint_scores.argmin(dim=0)
    soft_alpha = torch.gather(
        soft_hint_stack,
        0,
        soft_hint_winner.unsqueeze(0),
    ).squeeze(0)
    soft_alpha = torch.where(
        soft_hint_active & ~hard_hint_active,
        soft_alpha,
        torch.zeros_like(soft_alpha),
    )
    soft_owner = (
        torch.arange(len(ordered), device=device).view(-1, 1, 1)
        == soft_hint_winner.unsqueeze(0)
    ).to(dtype)
    effective = effective * (1.0 - soft_alpha.unsqueeze(0)) + soft_owner * soft_alpha.unsqueeze(0)
    hint_targets = hard_hint_targets + soft_owner * soft_alpha.unsqueeze(0)
    return raw_stack, effective, hint_targets


def build_character_masks(
    layout,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
    boundary_falloff: int = 0,
    body_expand: float = 0.0,
):
    """Return the historical two-value Body and ownership mask contract."""
    raw, effective, _hint_targets = build_character_mask_components(
        layout,
        height,
        width,
        device=device,
        dtype=dtype,
        boundary_falloff=boundary_falloff,
        body_expand=body_expand,
    )
    return raw, effective


def build_identity_core_masks(
    layout,
    height: int,
    width: int,
    radius: float,
    device=None,
    dtype=torch.float32,
):
    """Return smooth body-interior weights without changing ownership.

    ``radius`` is the central fraction of each Body box that receives a
    non-zero core weight. Ownership Hints are intentionally excluded so small
    contact boxes keep the normal soft regional blend.
    """
    device = device or torch.device("cpu")
    radius = max(0.05, min(1.0, float(radius)))
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cores = {
        item["uuid"]: torch.zeros((height, width), device=device, dtype=dtype)
        for item in layout["characters"]
    }
    threshold = 1.0 - radius
    for region in layout["regions"]:
        if not region["enabled"] or region["type"] != "body_region":
            continue
        half_width = max(float(region["width"]) * 0.5, 1e-6)
        half_height = max(float(region["height"]) * 0.5, 1e-6)
        center_x = float(region["x"]) + half_width
        center_y = float(region["y"]) + half_height
        distance = torch.maximum(
            (xx - center_x).abs() / half_width,
            (yy - center_y).abs() / half_height,
        )
        edge_depth = (1.0 - distance).clamp(0.0, 1.0)
        core = ((edge_depth - threshold) / radius).clamp(0.0, 1.0)
        core = core * core * (3.0 - 2.0 * core)
        key = region["character_uuid"]
        cores[key] = torch.maximum(cores[key], core.to(dtype=dtype))
    return torch.stack([cores[item["uuid"]] for item in layout["characters"]])


def render_inspection(layout, boundary_falloff: int = 0):
    height, width = layout["height"], layout["width"]
    _, effective = build_character_masks(
        layout,
        height,
        width,
        boundary_falloff=boundary_falloff,
    )
    raw_regions = build_raw_region_masks(layout, height, width)
    colors = torch.tensor(
        [[0.95, 0.24, 0.22], [0.10, 0.65, 0.92], [0.20, 0.78, 0.42], [0.96, 0.67, 0.13],
         [0.67, 0.35, 0.88], [0.93, 0.35, 0.68], [0.16, 0.76, 0.72], [0.75, 0.80, 0.22]],
        dtype=torch.float32,
    )
    image = torch.full((height, width, 3), 0.06, dtype=torch.float32)
    for index, mask in enumerate(effective):
        image += mask.unsqueeze(-1) * colors[index]
    raw_image = raw_regions.clamp(0.0, 1.0).unsqueeze(-1).expand(-1, -1, -1, 3)
    effective_image = effective.clamp(0.0, 1.0).unsqueeze(-1).expand(-1, -1, -1, 3)
    return image.clamp(0.0, 1.0).unsqueeze(0), effective, raw_image, effective_image
