import copy
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from anima_regional.masks import (
    build_raw_region_weights,
    build_region_weights,
    infer_grid_shape,
)
from anima_regional.nodes import (
    AnimaRegionalApply,
    AnimaRegionalCharacterBox,
    AnimaRegionalOptions,
    AnimaRegionalPreview,
)
from anima_regional.runtime import (
    RegionalCrossAttention,
    build_self_attention_allow_mask,
)
from anima_regional.v2.runtime import (
    EXTENDED_DELTA_NORM_CAP,
    _cap_residual_delta,
    _composition_profile,
    _detail_preserve_scale,
    _identity_detail_scale,
    _multi_character_profile,
    _scale_extended_delta,
    _shape_region_weights,
)
from anima_regional.v2.masks import build_identity_core_masks


def _box(prompt, center_x, priority=0):
    return {
        "geometry": "box",
        "prompt": prompt,
        "center_x": center_x,
        "center_y": 0.5,
        "width": 0.8,
        "height": 1.0,
        "feather": 0.0,
        "strength": 1.0,
        "priority": priority,
    }


class TestMasks(unittest.TestCase):
    def test_grid_uses_anima_patch_order(self):
        self.assertEqual(infer_grid_shape(32, (1, 4, 8, 16), 2, 1), (1, 4, 8))

    def test_exclusive_overlap_has_one_owner(self):
        regions = [_box("left", 0.35, priority=0), _box("right", 0.65, priority=1)]
        weights = build_region_weights(
            regions, 1, 1, 4, 4, torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.all(weights.sum(dim=0) <= 1.0 + 1e-6))
        overlap_columns = weights[:, 0, 1:3, 0]
        self.assertTrue(torch.all(overlap_columns[0] == 0.0))
        self.assertTrue(torch.all(overlap_columns[1] > 0.0))

    def test_normalized_overlap_is_explicitly_opt_in(self):
        regions = [_box("left", 0.5), _box("right", 0.5)]
        weights = build_region_weights(
            regions,
            1,
            1,
            2,
            2,
            torch.device("cpu"),
            torch.float32,
            overlap_mode="normalized",
        )
        self.assertTrue(torch.allclose(weights[:, 0, :, 0], torch.full((2, 4), 0.5)))

    def test_equal_priority_overlap_splits_by_anchor(self):
        regions = [_box("left", 0.3), _box("right", 0.7)]
        weights = build_region_weights(
            regions, 1, 1, 1, 10, torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.all(weights.sum(dim=0) <= 1.0 + 1e-6))
        self.assertTrue(torch.all(weights[0, 0, 3:5, 0] > 0.0))
        self.assertTrue(torch.all(weights[1, 0, 3:5, 0] == 0.0))
        self.assertTrue(torch.all(weights[1, 0, 5:7, 0] > 0.0))
        self.assertTrue(torch.all(weights[0, 0, 5:7, 0] == 0.0))

    def test_raw_box_masks_keep_both_sides_of_overlap(self):
        regions = [_box("left", 0.3), _box("right", 0.7)]
        raw = build_raw_region_weights(
            regions, 1, 1, 1, 10, torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.all(raw[:, 0, 3:7, 0] > 0.0))

    def test_boundary_falloff_zero_is_exactly_unchanged(self):
        regions = [_box("left", 0.3), _box("right", 0.7)]
        baseline = build_region_weights(
            regions, 1, 1, 2, 12, torch.device("cpu"), torch.float32
        )
        explicit_zero = build_region_weights(
            regions,
            1,
            1,
            2,
            12,
            torch.device("cpu"),
            torch.float32,
            boundary_falloff=0,
        )
        self.assertTrue(torch.equal(baseline, explicit_zero))

    def test_boundary_falloff_attenuates_only_internal_winners(self):
        regions = [
            dict(_box("left", 0.4), width=0.2),
            dict(_box("right", 0.6), width=0.2),
        ]
        baseline = build_region_weights(
            regions, 1, 1, 1, 20, torch.device("cpu"), torch.float32
        )
        softened = build_region_weights(
            regions,
            1,
            1,
            1,
            20,
            torch.device("cpu"),
            torch.float32,
            boundary_falloff=2,
        )
        base = baseline[:, 0, :, 0]
        soft = softened[:, 0, :, 0]
        self.assertTrue(torch.all((soft > 0.0).sum(dim=0) <= 1))
        changed = (soft + 1e-6) < base
        self.assertTrue(torch.any(changed))
        self.assertTrue(torch.any(changed[:, 8:12]))
        self.assertFalse(torch.any(changed[:, [6, 13]]))
        self.assertEqual(float(soft[0, 6]), 1.0)
        self.assertEqual(float(soft[1, 13]), 1.0)
        self.assertLess(float(soft[0, 9]), 1.0)
        self.assertLess(float(soft[1, 10]), 1.0)

    def test_boundary_falloff_does_not_activate_losing_character(self):
        regions = [_box("left", 0.5, priority=1), _box("right", 0.5, priority=0)]
        softened = build_region_weights(
            regions,
            1,
            1,
            2,
            8,
            torch.device("cpu"),
            torch.float32,
            boundary_falloff=2,
        )
        self.assertTrue(torch.all((softened[:, 0, :, 0] > 0.0).sum(dim=0) <= 1))


class FakeCrossAttention:
    is_selfattn = False
    q_proj = object()
    k_proj = object()
    v_proj = object()

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        value = context.mean(dim=(1, 2), keepdim=True)
        return value.expand_as(x)


class FakeDiffusionModel:
    patch_spatial = 2
    patch_temporal = 1

    def __init__(self, block_count=1):
        self.blocks = [type("Block", (), {"cross_attn": FakeCrossAttention()})() for _ in range(block_count)]

    def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
        return raw


class TestRuntime(unittest.TestCase):
    def _state(self, regions, blend_mode="replace"):
        return {
            "enabled": True,
            "regions": regions,
            "diffusion_model": FakeDiffusionModel(),
            "overlap_mode": "exclusive",
            "blend_mode": blend_mode,
            "global_strength": 1.0,
            "branch_chunk_size": 2,
            "patch_spatial": 2,
            "patch_temporal": 1,
            "sigma_range": None,
            "current_sigma": 1.0,
            "input_shape": (2, 4, 4, 8),
            "cond_or_uncond": [0, 1],
            "context_cache": {},
            "mask_cache": {},
            "disabled_layers": set(),
        }

    def test_region_routes_conditional_row_only(self):
        regions = [
            dict(_box("left", 0.25), conditioning={"raw": torch.full((1, 2, 3), 10.0), "ids": None, "weights": None}),
            dict(_box("right", 0.75), conditioning={"raw": torch.full((1, 2, 3), 20.0), "ids": None, "weights": None}),
        ]
        state = self._state(regions)
        original = state["diffusion_model"].blocks[0].cross_attn.forward
        wrapper = RegionalCrossAttention(
            original,
            state["diffusion_model"].blocks[0].cross_attn,
            state,
            0,
        )
        x = torch.zeros(2, 8, 3)
        context = torch.zeros(2, 2, 3)
        output = wrapper.forward(x, context, transformer_options={"cond_or_uncond": [0, 1]})
        self.assertTrue(torch.all(output[1] == 0.0))
        self.assertTrue(torch.any(output[0] == 10.0))
        self.assertTrue(torch.any(output[0] == 20.0))

    def test_exclusive_overlap_does_not_average_characters(self):
        regions = [
            dict(_box("a", 0.5, priority=1), conditioning={"raw": torch.full((1, 2, 3), 10.0), "ids": None, "weights": None}),
            dict(_box("b", 0.5, priority=0), conditioning={"raw": torch.full((1, 2, 3), 20.0), "ids": None, "weights": None}),
        ]
        state = self._state(regions)
        module = state["diffusion_model"].blocks[0].cross_attn
        wrapper = RegionalCrossAttention(module.forward, module, state, 0)
        output = wrapper.forward(
            torch.zeros(1, 8, 3),
            torch.zeros(1, 2, 3),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.all(output == 10.0))

    def test_strict_self_attention_blocks_cross_character_edges(self):
        left = dict(_box("left", 1.0 / 6.0), width=0.32)
        right = dict(_box("right", 5.0 / 6.0), width=0.32)
        state = {
            "regions": [left, right],
            "overlap_mode": "exclusive",
            "input_shape": (2, 4, 2, 6),
            "patch_spatial": 2,
            "patch_temporal": 1,
            "mask_cache": {},
        }
        x = torch.zeros(2, 3, 4)
        allowed = build_self_attention_allow_mask(
            state, x, x, [0, 1]
        )
        expected_cond = torch.tensor(
            [
                [True, True, False],
                [False, True, False],
                [False, True, True],
            ]
        )
        self.assertTrue(torch.equal(allowed[0, 0], expected_cond))
        self.assertTrue(torch.all(allowed[1, 0]))


class FakeClip:
    def tokenize(self, prompt):
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        marker = float((sum(ord(char) for char in str(tokens)) % 17) + 1)
        return [[
            torch.full((1, 2, 4), marker),
            {"t5xxl_ids": torch.tensor([1, 2])},
        ]]


class FakeSampling:
    def percent_to_sigma(self, value):
        return 1.0 - float(value)


class FakeModel:
    def __init__(self):
        self.diffusion_model = FakeDiffusionModel(block_count=3)
        self.model_options = {}
        self.object_patches = {}
        self.model_sampling = FakeSampling()

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.diffusion_model
        if name == "model_sampling":
            return self.model_sampling
        raise KeyError(name)

    def clone(self):
        cloned = copy.copy(self)
        cloned.model_options = dict(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        return cloned

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def add_object_patch(self, path, patch):
        self.object_patches[path] = patch


class TestNodes(unittest.TestCase):
    def test_chain_and_apply_register_expected_blocks(self):
        chain = AnimaRegionalCharacterBox().append(
            "red hair", 0.25, 0.5, 0.4, 0.8, 0.04, 1.0, 0
        )[0]
        chain = AnimaRegionalCharacterBox().append(
            "blue hair", 0.75, 0.5, 0.4, 0.8, 0.04, 1.0, 0, chain
        )[0]
        opts = AnimaRegionalOptions().build(
            "exclusive",
            "replace",
            "off",
            1.0,
            0,
            -1,
            0.0,
            1.0,
            2,
            8,
            -1,
            4096,
        )[0]
        model, positive, negative, status = AnimaRegionalApply().apply(
            FakeModel(),
            FakeClip(),
            chain,
            "anime scene",
            "low quality",
            True,
            opts,
        )
        self.assertEqual(len(model.object_patches), 3)
        self.assertEqual(len(positive), 1)
        self.assertEqual(len(negative), 1)
        self.assertIn("2 regions", status)

    def test_cross_attention_conflict_is_rejected(self):
        model = FakeModel()
        model.object_patches["diffusion_model.blocks.0.cross_attn.forward"] = object()
        chain = AnimaRegionalCharacterBox().append(
            "character", 0.5, 0.5, 0.5, 0.8, 0.04, 1.0, 0
        )[0]
        with self.assertRaises(ValueError):
            AnimaRegionalApply().apply(
                model,
                FakeClip(),
                chain,
                "scene",
                "",
                True,
                None,
            )

    def test_preview_returns_black_white_box_images(self):
        chain = AnimaRegionalCharacterBox().append(
            "left", 0.3, 0.5, 0.8, 1.0, 0.0, 1.0, 0
        )[0]
        chain = AnimaRegionalCharacterBox().append(
            "right", 0.7, 0.5, 0.8, 1.0, 0.0, 1.0, 0, chain
        )[0]
        preview, masks, box_images, effective_images = AnimaRegionalPreview().render(
            chain, 10, 4, "exclusive"
        )
        self.assertEqual(tuple(preview.shape), (1, 4, 10, 3))
        self.assertEqual(tuple(masks.shape), (2, 4, 10))
        self.assertEqual(tuple(box_images.shape), (2, 4, 10, 3))
        self.assertEqual(tuple(effective_images.shape), (2, 4, 10, 3))
        self.assertTrue(torch.all(box_images[:, 1, 3:7, :] > 0.0))
        self.assertTrue(torch.all(effective_images.sum(dim=0) <= 3.0 + 1e-6))

        _, _, selected_raw, selected_effective = AnimaRegionalPreview().render(
            chain, 10, 4, "exclusive", 2
        )
        self.assertEqual(tuple(selected_raw.shape), (1, 4, 10, 3))
        self.assertEqual(tuple(selected_effective.shape), (1, 4, 10, 3))

    def test_preview_effective_image_matches_boundary_falloff(self):
        chain = AnimaRegionalCharacterBox().append(
            "left", 0.4, 0.5, 0.2, 1.0, 0.0, 1.0, 0
        )[0]
        chain = AnimaRegionalCharacterBox().append(
            "right", 0.6, 0.5, 0.2, 1.0, 0.0, 1.0, 0, chain
        )[0]
        _, masks_zero, raw_zero, effective_zero = AnimaRegionalPreview().render(
            chain, 320, 64, "exclusive", 0, 0
        )
        _, masks_two, raw_two, effective_two = AnimaRegionalPreview().render(
            chain, 320, 64, "exclusive", 0, 2
        )
        self.assertTrue(torch.equal(raw_zero, raw_two))
        self.assertTrue(torch.equal(masks_zero.unsqueeze(-1).expand_as(effective_zero), effective_zero))
        self.assertTrue(torch.equal(masks_two.unsqueeze(-1).expand_as(effective_two), effective_two))
        self.assertTrue(torch.all(effective_two <= effective_zero + 1e-6))
        self.assertTrue(torch.any(effective_two < effective_zero - 1e-6))

    def test_boundary_controls_are_compatibility_safe_optionals(self):
        options = AnimaRegionalOptions.INPUT_TYPES()
        preview = AnimaRegionalPreview.INPUT_TYPES()
        self.assertIn("boundary_falloff", options["optional"])
        self.assertEqual(
            options["optional"]["composition_mode"][0],
            ["off", "early_layout"],
        )
        self.assertIn("composition_strength", options["optional"])
        self.assertIn("composition_expand", options["optional"])
        self.assertIn("composition_end_percent", options["optional"])
        self.assertEqual(
            options["optional"]["multi_character_guard"][0],
            ["off", "soft", "strong"],
        )
        self.assertEqual(
            options["optional"]["detail_preserve_mode"][0],
            ["off", "soft", "strong"],
        )
        self.assertIn("detail_preserve_start", options["optional"])
        self.assertIn("detail_preserve_amount", options["optional"])
        self.assertIn("edge_focus_power", options["optional"])
        self.assertEqual(
            options["optional"]["identity_anchor_mode"][0],
            ["off", "shared_delta"],
        )
        self.assertIn("identity_anchor_strength", options["optional"])
        self.assertIn("identity_late_floor", options["optional"])
        self.assertEqual(
            options["optional"]["identity_detail_mode"][0],
            ["off", "late"],
        )
        self.assertIn("identity_detail_start", options["optional"])
        self.assertIn("identity_detail_strength", options["optional"])
        self.assertIn("identity_core_strength", options["optional"])
        self.assertIn("identity_core_radius", options["optional"])
        self.assertEqual(
            list(preview["optional"]),
            ["selected_region", "boundary_falloff"],
        )

    def test_advanced_options_new_fields_are_optional_and_validated(self):
        baseline = AnimaRegionalOptions().build(
            "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8, -1, 4096
        )[0]
        self.assertEqual(baseline["composition_mode"], "off")
        self.assertEqual(baseline["composition_strength"], 1.0)
        configured = AnimaRegionalOptions().build(
            "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8, -1,
            4096, 0, "early_layout", 1.5, 0.08, 0.6,
        )[0]
        self.assertEqual(configured["composition_mode"], "early_layout")
        self.assertEqual(configured["composition_strength"], 1.5)
        self.assertEqual(configured["composition_expand"], 0.08)
        self.assertEqual(configured["composition_end_percent"], 0.6)
        with self.assertRaisesRegex(ValueError, "composition_expand"):
            AnimaRegionalOptions().build(
                "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8,
                -1, 4096, 0, "early_layout", 1.0, 0.2, 0.55,
            )
        guarded = AnimaRegionalOptions().build(
            "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8, -1,
            4096, 0, "off", 1.0, 0.04, 0.55, "soft",
        )[0]
        self.assertEqual(guarded["multi_character_guard"], "soft")
        with self.assertRaisesRegex(ValueError, "multi_character_guard"):
            AnimaRegionalOptions().build(
                "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8,
                -1, 4096, 0, "off", 1.0, 0.04, 0.55, "invalid",
            )
        protected = AnimaRegionalOptions().build(
            "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8, -1,
            4096, 0, "off", 1.0, 0.04, 0.55, "soft", "soft", 0.65, 0.5, 1.5,
            "shared_delta", 0.9, 0.75,
        )[0]
        self.assertEqual(protected["detail_preserve_mode"], "soft")
        self.assertEqual(protected["edge_focus_power"], 1.5)
        self.assertEqual(protected["identity_anchor_mode"], "shared_delta")
        self.assertEqual(protected["identity_anchor_strength"], 0.9)
        self.assertEqual(protected["identity_late_floor"], 0.75)
        identity = AnimaRegionalOptions().build(
            "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8, -1,
            4096, 0, "off", 1.0, 0.04, 0.55, "soft", "soft", 0.65,
            0.5, 1.5, "shared_delta", 0.9, 0.75, "late", 0.55, 0.8, 1.75, 0.5,
        )[0]
        self.assertEqual(identity["identity_detail_mode"], "late")
        self.assertEqual(identity["identity_detail_start"], 0.55)
        self.assertEqual(identity["identity_detail_strength"], 0.8)
        self.assertEqual(identity["identity_core_strength"], 1.75)
        self.assertEqual(identity["identity_core_radius"], 0.5)
        with self.assertRaisesRegex(ValueError, "identity_anchor_mode"):
            AnimaRegionalOptions().build(
                "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8,
                -1, 4096, 0, "off", 1.0, 0.04, 0.55, "off", "off", 0.65,
                0.5, 1.0, "invalid",
            )
        with self.assertRaisesRegex(ValueError, "identity_anchor_strength"):
            AnimaRegionalOptions().build(
                "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8,
                -1, 4096, 0, "off", 1.0, 0.04, 0.55, "off", "off", 0.65,
                0.5, 1.0, "shared_delta", 1.1,
            )
        with self.assertRaisesRegex(ValueError, "identity_late_floor"):
            AnimaRegionalOptions().build(
                "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8,
                -1, 4096, 0, "off", 1.0, 0.04, 0.55, "off", "off", 0.65,
                0.5, 1.0, "shared_delta", 1.0, -0.1,
            )
        with self.assertRaisesRegex(ValueError, "identity_detail_mode"):
            AnimaRegionalOptions().build(
                "exclusive", "replace", "off", 1.0, 0, -1, 0.0, 1.0, 2, 8,
                -1, 4096, 0, "off", 1.0, 0.04, 0.55, "off", "off", 0.65,
                0.5, 1.0, "off", 1.0, 0.8, "invalid",
            )

    def test_early_layout_profile_enhances_only_the_early_phase(self):
        state = {
            "composition_mode": "early_layout",
            "composition_strength": 1.0,
            "composition_expand": 0.08,
            "composition_end_percent": 0.55,
            "composition_sigma_range": (0.0, 1.0),
            "current_sigma": 0.9,
        }
        early_multiplier, early_expand = _composition_profile(state)
        self.assertGreater(early_multiplier, 1.0)
        self.assertGreater(early_expand, 0.0)
        state["current_sigma"] = 0.4
        late_multiplier, late_expand = _composition_profile(state)
        self.assertEqual((late_multiplier, late_expand), (1.0, 0.0))

    def test_extended_strength_preserves_baseline_and_caps_new_range(self):
        base = torch.ones(1, 2, 4)
        delta = torch.full_like(base, 0.5)
        self.assertTrue(torch.equal(_scale_extended_delta(delta, base, 2.0), delta * 2.0))
        large_delta = torch.full_like(base, 10.0)
        capped = _scale_extended_delta(large_delta, base, 4.0)
        cap = base.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt() * EXTENDED_DELTA_NORM_CAP
        norm = capped.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt()
        self.assertTrue(torch.all(norm <= cap + 1e-5))
        self.assertTrue(torch.any(capped < large_delta * 4.0))

    def test_multi_character_guard_scales_only_three_or_more_branches(self):
        self.assertEqual(_multi_character_profile({"multi_character_guard": "soft", "characters": [1, 2]}), (1.0, None))
        soft_scale, soft_cap = _multi_character_profile({"multi_character_guard": "soft", "characters": [1, 2, 3, 4]})
        strong_scale, strong_cap = _multi_character_profile({"multi_character_guard": "strong", "characters": [1, 2, 3, 4]})
        self.assertAlmostEqual(soft_scale, 2.0 ** -0.5)
        self.assertEqual(soft_cap, 2.0)
        self.assertEqual(strong_scale, 0.5)
        self.assertEqual(strong_cap, 1.5)

    def test_multi_character_residual_budget_caps_aggregate_delta(self):
        base = torch.ones(1, 2, 4)
        delta = torch.full_like(base, 10.0)
        capped = _cap_residual_delta(delta, base, 2.0)
        norm = capped.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt()
        limit = base.to(torch.float32).square().sum(dim=-1, keepdim=True).sqrt() * 2.0
        self.assertTrue(torch.all(norm <= limit + 1e-5))
        self.assertTrue(torch.any(capped < delta))

    def test_detail_preserve_is_identity_until_late_sampling(self):
        state = {
            "detail_preserve_mode": "soft",
            "detail_preserve_start": 0.65,
            "detail_preserve_amount": 0.5,
            "detail_sigma_range": (0.0, 1.0),
            "current_sigma": 0.8,
        }
        self.assertEqual(_detail_preserve_scale(state), 1.0)
        state["current_sigma"] = 0.0
        self.assertAlmostEqual(_detail_preserve_scale(state), 0.5)
        state["detail_preserve_mode"] = "strong"
        self.assertAlmostEqual(_detail_preserve_scale(state), 0.3)

    def test_detail_preserve_falls_back_without_sampling_metadata(self):
        state = {
            "detail_preserve_mode": "strong",
            "detail_preserve_start": 0.5,
            "detail_preserve_amount": 0.8,
            "detail_sigma_range": None,
            "current_sigma": 0.0,
        }
        self.assertEqual(_detail_preserve_scale(state), 1.0)

    def test_identity_detail_schedule_is_zero_before_start_and_full_at_end(self):
        state = {
            "identity_detail_mode": "late",
            "identity_detail_start": 0.6,
            "identity_detail_strength": 0.8,
            "sampling_sigma_range": (0.0, 1.0),
            "current_sigma": 0.8,
        }
        self.assertEqual(_identity_detail_scale(state), 0.0)
        state["current_sigma"] = 0.0
        self.assertAlmostEqual(_identity_detail_scale(state), 0.8)
        state["identity_detail_mode"] = "off"
        self.assertEqual(_identity_detail_scale(state), 0.0)

    def test_identity_core_weights_peak_in_body_center_and_ignore_hints(self):
        layout = {
            "characters": [{"uuid": "alice"}],
            "regions": [
                {
                    "character_uuid": "alice",
                    "type": "body_region",
                    "x": 0.2,
                    "y": 0.2,
                    "width": 0.6,
                    "height": 0.6,
                    "enabled": True,
                },
                {
                    "character_uuid": "alice",
                    "type": "ownership_hint",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.2,
                    "height": 0.2,
                    "enabled": True,
                },
            ],
        }
        core = build_identity_core_masks(
            layout,
            10,
            10,
            0.5,
            torch.device("cpu"),
            torch.float32,
        )[0]
        self.assertGreater(float(core[5, 5]), float(core[2, 2]))
        self.assertEqual(float(core[0, 0]), 0.0)

    def test_edge_focus_reduces_only_soft_weights(self):
        weights = torch.tensor([0.0, 0.25, 0.5, 1.0])
        shaped = _shape_region_weights(weights, 2.0)
        self.assertTrue(torch.equal(shaped, torch.tensor([0.0, 0.0625, 0.25, 1.0])))
        self.assertTrue(torch.equal(_shape_region_weights(weights, 1.0), weights))


if __name__ == "__main__":
    unittest.main()
