import copy
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from anima_regional import NODE_CLASS_MAPPINGS
from anima_regional.v2.masks import build_character_masks
from anima_regional.v2.nodes import (
    AnimaRegionalApplyV2,
    AnimaRegionalCharacterPromptV2,
    AnimaRegionalInspectV2,
    AnimaRegionalLayoutV2,
    AnimaRegionalPromptPackV2,
    AnimaRegionalSharedPromptV2,
)


class FakeClip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, prompt):
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        return [[torch.ones(1, 2, 4), {"t5xxl_ids": torch.tensor([1, 2])}]]


class FakeCrossAttention:
    q_proj = object()
    k_proj = object()
    v_proj = object()

    def __init__(self):
        self.context_means = []

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        if context is None:
            self.context_means.append(None)
            return x
        value = context.mean(dim=(1, 2), keepdim=True)
        self.context_means.append(float(value.flatten()[0]))
        return value.expand_as(x)


class FakeDiffusionModel:
    patch_spatial = 2
    patch_temporal = 1

    def __init__(self):
        self.blocks = [type("Block", (), {"cross_attn": FakeCrossAttention()})()]

    def preprocess_text_embeds(self, raw, ids, t5xxl_weights=None):
        return raw


class FakeModel:
    def __init__(self):
        self.diffusion_model = FakeDiffusionModel()
        self.model_options = {}
        self.object_patches = {}

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.diffusion_model
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


def character(identifier, label):
    return AnimaRegionalCharacterPromptV2().build(label, f"{label} prompt", 1.0, "#12AB34", identifier)[0]


def layout(characters, regions, mode="exclusive"):
    kwargs = {f"character_{index + 1}": value for index, value in enumerate(characters)}
    return AnimaRegionalLayoutV2().build(128, 64, mode, {"version": 2, "regions": regions}, **kwargs)[0]


def region(identifier, character_uuid, left, width, region_type="body_region"):
    return {
        "uuid": identifier,
        "character_uuid": character_uuid,
        "type": region_type,
        "geometry": "box",
        "x": left,
        "y": 0,
        "width": width,
        "height": 1,
        "feather": 0,
        "enabled": True,
    }


def conditioning(value=1.0):
    return [[torch.full((1, 2, 4), value), {"t5xxl_ids": torch.tensor([1, 2])}]]


class TestV2DataContract(unittest.TestCase):
    def test_registration_preserves_legacy_and_adds_frozen_v2_ids(self):
        self.assertIn("AnimaRegionalApply", NODE_CLASS_MAPPINGS)
        self.assertIs(NODE_CLASS_MAPPINGS["AnimaRegionalApplyV2"], AnimaRegionalApplyV2)
        self.assertIs(NODE_CLASS_MAPPINGS["AnimaRegionalSharedPromptV2"], AnimaRegionalSharedPromptV2)
        self.assertEqual(AnimaRegionalLayoutV2.RETURN_TYPES, ("ANIMA_REGIONAL_LAYOUT_V2", "INT", "INT"))
        self.assertEqual(
            AnimaRegionalApplyV2.INPUT_TYPES()["optional"]["advanced_options"][0],
            "ANIMA_REGIONAL_OPTIONS",
        )

    def test_shared_scene_prompt_is_a_fanout_safe_string(self):
        node = AnimaRegionalSharedPromptV2()
        self.assertEqual(node.emit("  two characters, touching hands  \n")[0], "  two characters, touching hands  \n")
        self.assertEqual(node.emit(None)[0], "")
        self.assertEqual(node.RETURN_TYPES, ("STRING",))

    def test_hidden_unique_id_is_stable_and_distinguishes_nodes(self):
        node = AnimaRegionalCharacterPromptV2()
        first = node.build("Alice", "prompt", 1.0, "", unique_id="42")[0]
        reload = node.build("Alice", "prompt", 1.0, "", unique_id="42")[0]
        duplicate = node.build("Alice", "prompt", 1.0, "", unique_id="43")[0]
        self.assertEqual(first["uuid"], reload["uuid"])
        self.assertNotEqual(first["uuid"], duplicate["uuid"])

    def test_character_strength_supports_protected_extended_range(self):
        node = AnimaRegionalCharacterPromptV2()
        payload = node.build("Alice", "prompt", 4.0, "", unique_id="42")[0]
        self.assertEqual(payload["strength"], 4.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 4"):
            node.build("Alice", "prompt", 4.1, "", unique_id="42")

    def test_reordered_character_inputs_preserve_uuid_bindings(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        editor = {
            "version": 2,
            "characters": [alice, bob],
            "regions": [region("a", "alice", 0.0, 0.5), region("b", "bob", 0.5, 0.5)],
        }
        result = AnimaRegionalLayoutV2().build(
            128,
            64,
            "exclusive",
            editor,
            character_1=bob,
            character_2=alice,
        )[0]
        self.assertEqual([item["uuid"] for item in result["characters"]], ["alice", "bob"])
        self.assertEqual([item["character_uuid"] for item in result["regions"]], ["alice", "bob"])

    def test_layout_rejects_unknown_character_and_clamps_stale_coordinates(self):
        alice = character("alice", "Alice")
        with self.assertRaisesRegex(ValueError, "unknown character"):
            layout([alice], [{"uuid": "r1", "character_uuid": "bob", "type": "body_region", "geometry": "box", "x": 0, "y": 0, "width": .5, "height": 1, "feather": 0, "enabled": True}])
        result = layout([alice], [{"uuid": "r1", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": -.2, "y": .8, "width": 2, "height": 2, "feather": -.1, "enabled": True}])
        self.assertEqual(result["regions"][0]["x"], 0.0)
        self.assertAlmostEqual(result["regions"][0]["height"], .2)
        self.assertEqual(result["regions"][0]["feather"], 0.0)

    def test_character_is_encoded_once_even_with_two_body_regions(self):
        alice = character("alice", "Alice")
        regions = [
            {"uuid": "r1", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": 0, "y": 0, "width": .5, "height": 1, "feather": 0, "enabled": True},
            {"uuid": "r2", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": .5, "y": 0, "width": .5, "height": 1, "feather": 0, "enabled": True},
        ]
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(clip, layout([alice], regions), "global", "negative")[0]
        self.assertEqual(len(pack["characters"]), 1)
        self.assertEqual(clip.encoded, ["global", "negative", "Alice prompt\nglobal"])
        self.assertTrue(pack["shared_is_final"])
        self.assertIs(pack["shared_conditioning"]["raw"], pack["positive"][0][0])

    def test_external_conditioning_remains_final_while_shared_scene_is_encoded(self):
        alice = character("alice", "Alice")
        external_positive, external_negative = conditioning(3.0), conditioning(-1.0)
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(clip, layout([alice], []), "global", "negative", external_positive, external_negative)[0]
        model, positive, negative, status = AnimaRegionalApplyV2().apply(FakeModel(), pack, False, "replace")
        self.assertIs(positive, external_positive)
        self.assertIs(negative, external_negative)
        self.assertEqual(status, "disabled; original model returned")
        self.assertEqual(clip.encoded, ["global", "Alice prompt\nglobal"])
        self.assertFalse(pack["shared_is_final"])
        self.assertIsNot(pack["shared_conditioning"]["raw"], external_positive[0][0])

    def test_invalid_conditioning_is_rejected_before_runtime(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(), layout([alice], [region("a", "alice", 0.0, 1.0)]), "", ""
        )[0]
        broken_character = copy.deepcopy(pack)
        broken_character["characters"][0]["conditioning"]["raw"] = object()
        with self.assertRaisesRegex(ValueError, "no tensor embedding"):
            AnimaRegionalApplyV2().apply(FakeModel(), broken_character, False, "replace")
        broken_final = copy.deepcopy(pack)
        broken_final["positive"] = object()
        with self.assertRaisesRegex(ValueError, "final positive conditioning"):
            AnimaRegionalApplyV2().apply(FakeModel(), broken_final, False, "replace")
        broken_shared = copy.deepcopy(pack)
        broken_shared["shared_conditioning"]["raw"] = object()
        with self.assertRaisesRegex(ValueError, "shared conditioning"):
            AnimaRegionalApplyV2().apply(FakeModel(), broken_shared, False, "replace")


class TestV2MasksAndApply(unittest.TestCase):
    def test_hint_outside_body_locally_activates_only_its_box(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.25),
            region("b", "bob", 0.75, 0.25),
            region("hint", "bob", 0.25, 0.25, "ownership_hint"),
        ]
        raw, exclusive = build_character_masks(layout([alice, bob], regions), 1, 8)
        self.assertTrue(torch.equal(raw[1, 0], torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.])))
        self.assertTrue(torch.equal(exclusive[1, 0], torch.tensor([0., 0., 1., 1., 0., 0., 1., 1.])))

        _, normalized = build_character_masks(layout([alice, bob], regions, "normalized"), 1, 8)
        self.assertTrue(torch.equal(normalized[1, 0], raw[1, 0]))

    def test_boundary_falloff_zero_is_identity_and_nonzero_softens_only_boundary(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [region("a", "alice", 0.0, 0.5), region("b", "bob", 0.5, 0.5)]
        item = layout([alice, bob], regions)
        _, default = build_character_masks(item, 1, 8)
        _, zero = build_character_masks(item, 1, 8, boundary_falloff=0)
        _, softened = build_character_masks(item, 1, 8, boundary_falloff=1)
        self.assertTrue(torch.equal(default, zero))
        self.assertEqual(float(softened[0, 0, 0]), 1.0)
        self.assertLess(float(softened[0, 0, 3]), 1.0)
        self.assertEqual(float(softened[1, 0, 7]), 1.0)
        self.assertLess(float(softened[1, 0, 4]), 1.0)
        self.assertTrue(torch.all(softened.sum(dim=0) <= 1.0))

    def test_inspect_preserves_public_names_and_raw_region_batch(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        item = layout(
            [alice, bob],
            [
                region("a1", "alice", 0.0, 0.25),
                region("a2", "alice", 0.25, 0.25),
                region("hint", "bob", 0.5, 0.25, "ownership_hint"),
            ],
        )
        self.assertEqual(
            AnimaRegionalInspectV2.RETURN_NAMES,
            ("ownership_preview", "region_masks", "box_masks_image", "effective_masks_image"),
        )
        ownership, masks, boxes, effective = AnimaRegionalInspectV2().render(item)
        self.assertEqual(tuple(ownership.shape), (1, 64, 128, 3))
        self.assertEqual(tuple(masks.shape), (2, 64, 128))
        self.assertEqual(tuple(boxes.shape), (3, 64, 128, 3))
        self.assertEqual(tuple(effective.shape), (2, 64, 128, 3))

    def test_body_regions_max_union_and_hints_do_not_expand_body(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            {"uuid": "a1", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": 0, "y": 0, "width": .75, "height": 1, "feather": 0, "enabled": True},
            {"uuid": "a2", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": .5, "y": 0, "width": .25, "height": 1, "feather": 0, "enabled": True},
            {"uuid": "b1", "character_uuid": "bob", "type": "body_region", "geometry": "box", "x": .25, "y": 0, "width": .75, "height": 1, "feather": 0, "enabled": True},
            {"uuid": "hint", "character_uuid": "bob", "type": "ownership_hint", "geometry": "box", "x": .25, "y": 0, "width": .25, "height": 1, "feather": 0, "enabled": True},
        ]
        raw, exclusive = build_character_masks(layout([alice, bob], regions), 1, 8)
        self.assertTrue(torch.all(raw[0, 0, :6] == 1))
        self.assertGreaterEqual(int((raw[1, 0] > 0).sum()), 5)
        self.assertEqual(float(raw[1, 0, 0]), 0.0)  # Hint never expands Bob's body.
        self.assertTrue(torch.all(exclusive.sum(dim=0) <= 1.0))
        _, normalized = build_character_masks(layout([alice, bob], regions, "normalized"), 1, 8)
        self.assertEqual(float(normalized[0, 0, 2]), 0.5)  # Hint is ignored in normalized mode.

    def test_early_body_expansion_does_not_expand_ownership_hint(self):
        alice = character("alice", "Alice")
        regions = [
            region("body", "alice", 0.4, 0.2),
            region("hint", "alice", 0.8, 0.1, "ownership_hint"),
        ]
        raw, effective = build_character_masks(
            layout([alice], regions), 1, 10, body_expand=0.2
        )
        self.assertTrue(torch.all(raw[0, 0, 2:8] == 1))
        self.assertEqual(float(raw[0, 0, 0]), 0.0)
        self.assertEqual(float(effective[0, 0, 0]), 0.0)

    def test_apply_rejects_unknown_cross_attention_patch(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(FakeClip(), layout([alice], [{"uuid": "r1", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": 0, "y": 0, "width": 1, "height": 1, "feather": 0, "enabled": True}]), "", "")[0]
        model = FakeModel()
        model.object_patches["diffusion_model.blocks.0.cross_attn.forward"] = lambda *args, **kwargs: None
        with self.assertRaisesRegex(ValueError, "unknown cross-attention"):
            AnimaRegionalApplyV2().apply(model, pack, True, "replace")

    def test_apply_registers_character_level_router(self):
        alice = character("alice", "Alice")
        region = {"uuid": "r1", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": 0, "y": 0, "width": 1, "height": 1, "feather": 0, "enabled": True}
        pack = AnimaRegionalPromptPackV2().pack(FakeClip(), layout([alice], [region]), "", "")[0]
        patched, positive, negative, status = AnimaRegionalApplyV2().apply(FakeModel(), pack, True, "replace")
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        self.assertTrue(getattr(patch, "_anima_regional_cross_attn_patch", False))
        self.assertIs(positive, pack["positive"])
        self.assertIs(negative, pack["negative"])
        self.assertIn("1 characters", status)
        self.assertIn("multi_guard=off", status)

    def test_apply_filters_inactive_branches_and_honors_global_strength(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(), layout([alice, bob], [region("a", "alice", 0.0, 1.0)]), "", ""
        )[0]
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            FakeModel(), pack, True, "replace", {
                "global_strength": 0.25,
                "boundary_falloff": 2,
                "multi_character_guard": "soft",
                "detail_preserve_mode": "soft",
                "detail_preserve_start": 0.65,
                "detail_preserve_amount": 0.5,
                "edge_focus_power": 1.5,
            }
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        state = patch.router.state
        self.assertEqual([item["uuid"] for item in state["characters"]], ["alice"])
        self.assertEqual(state["global_strength"], 0.25)
        self.assertEqual(state["boundary_falloff"], 2)
        self.assertEqual(state["multi_character_guard"], "soft")
        self.assertEqual(state["detail_preserve_mode"], "soft")
        self.assertEqual(state["edge_focus_power"], 1.5)
        state["input_shape"] = (1, 4, 4, 8)
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.zeros((1, 2, 4)),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 0.25)))

    def test_shared_delta_uses_character_minus_shared_scene(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
        )[0]
        pack["shared_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 13.0)
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            FakeModel(),
            pack,
            True,
            "replace",
            {
                "identity_anchor_mode": "shared_delta",
                "identity_anchor_strength": 1.0,
                "identity_late_floor": 0.8,
            },
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        patch.router.state["input_shape"] = (1, 4, 4, 8)
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 103.0)))

    def test_internal_positive_reuses_base_for_shared_delta(self):
        alice = character("alice", "Alice")
        model = FakeModel()
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
        )[0]
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {"identity_anchor_mode": "shared_delta"},
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        patch.router.state["input_shape"] = (1, 4, 4, 8)
        patch(
            torch.zeros((1, 8, 4)),
            torch.ones((1, 2, 4)),
            transformer_options={"cond_or_uncond": [0]},
        )
        calls = model.diffusion_model.blocks[0].cross_attn.context_means
        self.assertEqual(calls, [1.0, 1.0])
        self.assertEqual(patch.router.state["shared_context_cache"], {})

    def test_external_positive_evaluates_one_shared_branch_for_all_characters(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        model = FakeModel()
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout(
                [alice, bob],
                [
                    region("a", "alice", 0.0, 0.5),
                    region("b", "bob", 0.5, 0.5),
                ],
            ),
            "global",
            "",
            conditioning(100.0),
        )[0]
        pack["shared_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 11.0)
        pack["characters"][1]["conditioning"]["raw"] = torch.full((1, 2, 4), 12.0)
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {"identity_anchor_mode": "shared_delta"},
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        patch.router.state["input_shape"] = (1, 4, 4, 8)
        patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        calls = model.diffusion_model.blocks[0].cross_attn.context_means
        self.assertEqual(calls, [100.0, 10.0, 11.0, 12.0])

    def test_zero_identity_strength_is_exact_baseline_without_shared_call(self):
        alice = character("alice", "Alice")
        model = FakeModel()
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
        )[0]
        pack["shared_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 13.0)
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {
                "identity_anchor_mode": "shared_delta",
                "identity_anchor_strength": 0.0,
                "identity_late_floor": 0.8,
                "detail_preserve_mode": "strong",
                "detail_preserve_start": 0.5,
                "detail_preserve_amount": 0.8,
            },
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        state = patch.router.state
        state["input_shape"] = (1, 4, 4, 8)
        state["detail_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.0
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 82.6)))
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 13.0],
        )

    def test_shared_delta_broadcasts_context_and_preserves_runtime_dtype(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
        )[0]
        pack["shared_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 13.0)
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            FakeModel(),
            pack,
            True,
            "replace",
            {"identity_anchor_mode": "shared_delta"},
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        state = patch.router.state
        state["input_shape"] = (2, 4, 4, 8)
        context = torch.tensor(
            [100.0, 200.0],
            dtype=torch.bfloat16,
        ).view(2, 1, 1).expand(2, 2, 4)
        output = patch(
            torch.zeros((2, 8, 4), dtype=torch.bfloat16),
            context,
            transformer_options={"cond_or_uncond": [0, 1]},
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.all(output[0] == torch.tensor(103.0, dtype=torch.bfloat16)))
        self.assertTrue(torch.all(output[1] == torch.tensor(200.0, dtype=torch.bfloat16)))
        self.assertEqual(len(state["context_cache"]), 1)
        self.assertEqual(len(state["shared_context_cache"]), 1)

    def test_identity_late_floor_keeps_character_residual_alive(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
        )[0]
        pack["shared_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 13.0)
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            FakeModel(),
            pack,
            True,
            "replace",
            {
                "identity_anchor_mode": "shared_delta",
                "identity_late_floor": 0.8,
                "detail_preserve_mode": "strong",
                "detail_preserve_start": 0.5,
                "detail_preserve_amount": 0.8,
            },
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        state = patch.router.state
        state["input_shape"] = (1, 4, 4, 8)
        state["detail_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.0
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 102.4)))


if __name__ == "__main__":
    unittest.main()
