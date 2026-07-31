"""Optional cross-attention patch composition contracts.

This module deliberately knows no Mixer implementation details.  It accepts
only the public marker carried by the post-Adapter Anchor-Q callable.
"""

from __future__ import annotations


_ADAPTER_ANCHOR_Q_MARKER = "_anima_adapter_anchor_q_forward_patch"
_REGIONAL_MARKER = "_anima_regional_cross_attn_patch"
_ARTIST_CROSS_ATTN_MARKER = "_anima_artist_mixer_forward_patch"


def _patch_error(patch_path: str, reason: str) -> ValueError:
    return ValueError(
        f"[AnimaRegional] incompatible cross-attention patch at "
        f"{patch_path}: {reason}"
    )


def resolve_regional_inner_forward(existing_patches, patch_path, default_forward):
    """Return the callable Regional must invoke below its own router.

    A missing object patch preserves the 0.1.2 path.  The only composable
    patch is the Marker/callable protocol for post-Adapter Anchor-Q.  Every
    other patch type is rejected because its routing order is ambiguous.
    """
    patches = existing_patches or {}
    if patch_path not in patches:
        return default_forward, "native"

    patch = patches[patch_path]
    if not callable(patch):
        raise _patch_error(patch_path, "the patch is not callable")
    marker_states = {
        "Adapter Anchor-Q": getattr(patch, _ADAPTER_ANCHOR_Q_MARKER, False) is True,
        "Regional": getattr(patch, _REGIONAL_MARKER, False) is True,
        "full Artist Cross-Attn": getattr(patch, _ARTIST_CROSS_ATTN_MARKER, False) is True,
    }
    active_markers = [name for name, enabled in marker_states.items() if enabled]
    if len(active_markers) > 1:
        raise _patch_error(
            patch_path,
            f"ambiguous patch carries multiple protocol markers: {', '.join(active_markers)}",
        )
    if marker_states["Adapter Anchor-Q"]:
        return patch, "adapter_anchor_q"
    if marker_states["Regional"]:
        raise _patch_error(
            patch_path,
            "Regional is already applied; do not apply Regional twice or in reverse order",
        )
    if marker_states["full Artist Cross-Attn"]:
        raise _patch_error(
            patch_path,
            "the full Artist Cross-Attn Mixer cannot compose with Regional",
        )
    raise _patch_error(
        patch_path,
        "unknown cross-attention patches cannot compose with Regional",
    )
