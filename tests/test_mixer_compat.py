import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from anima_regional.compat import resolve_regional_inner_forward
from anima_regional.runtime import RegionalCrossAttention
from anima_regional.v2.runtime import V2RegionalCrossAttention


class _RegionalPatch:
    _anima_regional_cross_attn_patch = True

    def __call__(self, *args, **kwargs):
        return None


class _ArtistCrossAttentionPatch:
    _anima_artist_mixer_forward_patch = True

    def __call__(self, *args, **kwargs):
        return None


class _AnchorQPatch:
    _anima_adapter_anchor_q_forward_patch = True

    def __init__(self, offset=7.0):
        self.offset = offset
        self.calls = []
        self.transformed_q = []
        self.options = []

    def __call__(self, x, context=None, rope_emb=None, transformer_options=None):
        self.calls.append(x.detach().clone())
        self.options.append(transformer_options)
        q_input = x + self.offset
        self.transformed_q.append(q_input.detach().clone())
        context_value = context.mean(dim=(1, 2), keepdim=True)
        return q_input + context_value


class _CrossAttention:
    is_selfattn = False
    q_proj = object()
    k_proj = object()
    v_proj = object()

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        return x + context.mean(dim=(1, 2), keepdim=True)


class _AmbiguousAnchorPatch(_AnchorQPatch):
    _anima_regional_cross_attn_patch = True


class _DiffusionModel:
    patch_spatial = 2
    patch_temporal = 1

    def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
        return raw


def _region(center_x, value):
    return {
        "geometry": "box",
        "center_x": center_x,
        "center_y": 0.5,
        "width": 0.5,
        "height": 1.0,
        "feather": 0.0,
        "strength": 1.0,
        "conditioning": {
            "raw": torch.full((1, 2, 3), value),
            "ids": None,
            "weights": None,
        },
    }


def _runtime_state():
    return {
        "enabled": True,
        "regions": [_region(0.25, 10.0), _region(0.75, 20.0)],
        "diffusion_model": _DiffusionModel(),
        "overlap_mode": "exclusive",
        "blend_mode": "replace",
        "global_strength": 1.0,
        "branch_chunk_size": 2,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "sigma_range": None,
        "current_sigma": 1.0,
        "input_shape": (1, 4, 4, 8),
        "cond_or_uncond": [0],
        "context_cache": {},
        "mask_cache": {},
        "disabled_layers": set(),
        "q_reuse_failed": False,
        "q_reuse_validated": None,
        "inner_attention_kinds": {0: "adapter_anchor_q"},
    }


def _v2_runtime_state():
    return {
        "layout": {
            "version": 2,
            "width": 128,
            "height": 64,
            "overlap_mode": "exclusive",
            "characters": [
                {"uuid": "alice", "label": "Alice"},
                {"uuid": "bob", "label": "Bob"},
            ],
            "regions": [
                {
                    "uuid": "left",
                    "character_uuid": "alice",
                    "type": "body_region",
                    "geometry": "box",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.5,
                    "height": 1.0,
                    "feather": 0.0,
                    "enabled": True,
                },
                {
                    "uuid": "right",
                    "character_uuid": "bob",
                    "type": "body_region",
                    "geometry": "box",
                    "x": 0.5,
                    "y": 0.0,
                    "width": 0.5,
                    "height": 1.0,
                    "feather": 0.0,
                    "enabled": True,
                },
            ],
        },
        "characters": [
            {
                "uuid": "alice",
                "strength": 1.0,
                "conditioning": {
                    "raw": torch.full((1, 2, 3), 10.0),
                    "ids": None,
                    "weights": None,
                },
            },
            {
                "uuid": "bob",
                "strength": 1.0,
                "conditioning": {
                    "raw": torch.full((1, 2, 3), 20.0),
                    "ids": None,
                    "weights": None,
                },
            },
        ],
        "shared_conditioning": {
            "raw": torch.full((1, 2, 3), 5.0),
            "ids": None,
            "weights": None,
        },
        "shared_is_final": False,
        "diffusion_model": _DiffusionModel(),
        "blend_mode": "replace",
        "global_strength": 1.0,
        "boundary_falloff": 0,
        "composition_mode": "off",
        "multi_character_guard": "off",
        "detail_preserve_mode": "off",
        "edge_focus_power": 1.0,
        "identity_anchor_mode": "shared_delta",
        "identity_anchor_strength": 1.0,
        "identity_late_floor": 0.8,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "sigma_range": None,
        "current_sigma": 1.0,
        "input_shape": (1, 4, 4, 8),
        "cond_or_uncond": [0],
        "context_cache": {},
        "shared_context_cache": {},
        "mask_cache": {},
        "disabled_layers": set(),
    }


class TestMixerProtocol(unittest.TestCase):
    def test_missing_patch_preserves_native_forward(self):
        def native(*args, **kwargs):
            return args, kwargs

        resolved, kind = resolve_regional_inner_forward({}, "block.path", native)
        self.assertIs(resolved, native)
        self.assertEqual(kind, "native")

    def test_unknown_patch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown cross-attention"):
            resolve_regional_inner_forward(
                {"block.path": lambda *args, **kwargs: None},
                "block.path",
                lambda *args, **kwargs: None,
            )

    def test_known_incompatible_patches_are_rejected(self):
        for patch, expected in (
            (_RegionalPatch(), "already applied"),
            (_ArtistCrossAttentionPatch(), "full Artist Cross-Attn"),
            (object(), "not callable"),
        ):
            with self.subTest(patch=type(patch).__name__):
                with self.assertRaisesRegex(ValueError, expected):
                    resolve_regional_inner_forward(
                        {"block.path": patch},
                        "block.path",
                        lambda *args, **kwargs: None,
                    )

    def test_ambiguous_anchor_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous patch"):
            resolve_regional_inner_forward(
                {"block.path": _AmbiguousAnchorPatch()},
                "block.path",
                lambda *args, **kwargs: None,
            )

    def test_anchor_q_is_the_inner_forward_for_base_and_regions(self):
        anchor = _AnchorQPatch()
        state = _runtime_state()
        module = _CrossAttention()
        wrapper = RegionalCrossAttention(anchor, module, state, 0)
        x = torch.full((1, 8, 3), 2.0)
        context = torch.zeros((1, 2, 3))

        options = {"cond_or_uncond": [0], "sentinel": object()}
        output = wrapper.forward(x, context, transformer_options=options)

        self.assertEqual(len(anchor.calls), 3)
        self.assertTrue(all(torch.equal(call, x) for call in anchor.calls))
        expected_q = torch.full_like(x, 9.0)
        self.assertTrue(
            all(torch.equal(transformed, expected_q) for transformed in anchor.transformed_q)
        )
        self.assertTrue(all(value is options for value in anchor.options))
        expected = torch.tensor(
            [[19.0, 19.0, 29.0, 29.0, 19.0, 19.0, 29.0, 29.0]]
        ).unsqueeze(-1).expand_as(output)
        self.assertTrue(torch.equal(output, expected))

    def test_v2_identity_anchor_uses_anchor_q_for_shared_and_character_branches(self):
        anchor = _AnchorQPatch()
        state = _v2_runtime_state()
        wrapper = V2RegionalCrossAttention(anchor, state, 0)
        x = torch.full((1, 8, 3), 2.0)
        context = torch.zeros((1, 2, 3))
        options = {"cond_or_uncond": [0], "sentinel": object()}

        output = wrapper.forward(x, context, transformer_options=options)

        self.assertEqual(len(anchor.calls), 4)
        self.assertTrue(all(torch.equal(call, x) for call in anchor.calls))
        expected_q = torch.full_like(x, 9.0)
        self.assertTrue(
            all(torch.equal(transformed, expected_q) for transformed in anchor.transformed_q)
        )
        self.assertTrue(all(value is options for value in anchor.options))
        expected = torch.tensor(
            [[14.0, 14.0, 24.0, 24.0, 14.0, 14.0, 24.0, 24.0]]
        ).unsqueeze(-1).expand_as(output)
        self.assertTrue(torch.equal(output, expected))

    def test_v2_late_identity_detail_keeps_q_only_anchor_as_inner_callable(self):
        anchor = _AnchorQPatch()
        state = _v2_runtime_state()
        state["characters"][0]["identity_conditioning"] = {
            "raw": torch.full((1, 2, 3), 30.0),
            "ids": None,
            "weights": None,
        }
        state["has_identity_conditioning"] = True
        state["identity_detail_mode"] = "late"
        state["identity_detail_start"] = 0.0
        state["identity_detail_strength"] = 0.25
        state["sampling_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.0
        wrapper = V2RegionalCrossAttention(anchor, state, 0)
        x = torch.full((1, 8, 3), 2.0)
        context = torch.zeros((1, 2, 3))
        options = {"cond_or_uncond": [0], "sentinel": object()}

        output = wrapper.forward(x, context, transformer_options=options)

        self.assertEqual(len(anchor.calls), 5)
        self.assertTrue(all(torch.equal(call, x) for call in anchor.calls))
        expected_q = torch.full_like(x, 9.0)
        self.assertTrue(
            all(torch.equal(transformed, expected_q) for transformed in anchor.transformed_q)
        )
        self.assertTrue(all(value is options for value in anchor.options))
        self.assertEqual(output.shape, x.shape)

    def test_plain_runtime_path_is_identity_when_disabled(self):
        state = _runtime_state()
        state["enabled"] = False
        module = _CrossAttention()
        x = torch.randn((1, 8, 3))
        context = torch.randn((1, 2, 3))
        wrapper = RegionalCrossAttention(module.forward, module, state, 0)

        self.assertTrue(torch.equal(
            wrapper.forward(x, context), module.forward(x, context)
        ))


if __name__ == "__main__":
    unittest.main()
