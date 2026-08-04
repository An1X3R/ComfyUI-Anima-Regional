import copy
import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from anima_regional import NODE_CLASS_MAPPINGS
from anima_regional.v2.masks import (
    build_character_mask_components,
    build_character_masks,
)
from anima_regional.v2.nodes import (
    AnimaRegionalApplyV2,
    AnimaRegionalCharacterPromptV2,
    AnimaRegionalInspectV2,
    AnimaRegionalLayoutV2,
    AnimaRegionalOptionsV2,
    AnimaRegionalPromptCompilerV2,
    AnimaRegionalPromptPackV2,
    AnimaRegionalSharedPromptV2,
)
from anima_regional.v2.runtime import _adaptive_character_focus


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


def region(
    identifier,
    character_uuid,
    left,
    width,
    region_type="body_region",
    *,
    hint_blend=None,
    strength=None,
    priority=None,
):
    payload = {
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
    if hint_blend is not None:
        payload["hint_blend"] = hint_blend
    if strength is not None:
        payload["strength"] = strength
    if priority is not None:
        payload["priority"] = priority
    return payload


def conditioning(value=1.0):
    return [[torch.full((1, 2, 4), value), {"t5xxl_ids": torch.tensor([1, 2])}]]


class TestV2DataContract(unittest.TestCase):
    def test_registration_preserves_legacy_and_adds_frozen_v2_ids(self):
        self.assertIn("AnimaRegionalApply", NODE_CLASS_MAPPINGS)
        self.assertIs(NODE_CLASS_MAPPINGS["AnimaRegionalApplyV2"], AnimaRegionalApplyV2)
        self.assertIs(NODE_CLASS_MAPPINGS["AnimaRegionalSharedPromptV2"], AnimaRegionalSharedPromptV2)
        self.assertIs(
            NODE_CLASS_MAPPINGS["AnimaRegionalPromptCompilerV2"],
            AnimaRegionalPromptCompilerV2,
        )
        self.assertEqual(AnimaRegionalLayoutV2.RETURN_TYPES, ("ANIMA_REGIONAL_LAYOUT_V2", "INT", "INT"))
        self.assertEqual(
            AnimaRegionalApplyV2.INPUT_TYPES()["optional"]["advanced_options"][0],
            "ANIMA_REGIONAL_OPTIONS",
        )
        self.assertIs(NODE_CLASS_MAPPINGS["AnimaRegionalOptionsV2"], AnimaRegionalOptionsV2)

    def test_v2_options_expose_only_effective_routing_controls(self):
        inputs = AnimaRegionalOptionsV2.INPUT_TYPES()
        self.assertNotIn("self_attention_mode", inputs["required"])
        self.assertNotIn("branch_chunk_size", inputs["required"])
        self.assertIn("layout_strength", inputs["optional"])
        self.assertIn("hint_constraint_mode", inputs["optional"])
        self.assertEqual(
            list(inputs["optional"])[-2:],
            ["hint_constraint_mode", "character_focus_mode"],
        )
        options = AnimaRegionalOptionsV2().build(
            1.0,
            0,
            -1,
            0.0,
            1.0,
            layout_strength=1.25,
            hint_constraint_mode="soft",
            character_focus_mode="adaptive",
        )[0]
        self.assertEqual(options["options_contract"], "v2_routing_v1")
        self.assertEqual(options["layout_strength"], 1.25)
        self.assertEqual(options["hint_constraint_mode"], "soft")
        self.assertEqual(options["character_focus_mode"], "adaptive")
        with self.assertRaisesRegex(ValueError, "character_focus_mode"):
            AnimaRegionalOptionsV2().build(
                1.0,
                0,
                -1,
                0.0,
                1.0,
                character_focus_mode="invalid",
            )

    def test_shared_scene_prompt_is_a_fanout_safe_string(self):
        node = AnimaRegionalSharedPromptV2()
        self.assertEqual(node.emit("  two characters, touching hands  \n")[0], "  two characters, touching hands  \n")
        self.assertEqual(node.emit(None)[0], "")
        self.assertEqual(node.RETURN_TYPES, ("STRING",))

    def test_prompt_compiler_keeps_regional_and_mixer_text_in_sync(self):
        kaltsit = AnimaRegionalCharacterPromptV2().build(
            "Kal'tsit",
            "",
            1.0,
            "",
            "kaltsit",
            "Kal'tsit identity",
            "standing on the left, touching Mon3tr",
        )[0]
        mon3tr = AnimaRegionalCharacterPromptV2().build(
            "Mon3tr",
            "",
            1.0,
            "",
            "mon3tr",
            "Mon3tr identity",
            "standing on the right, being touched by Kal'tsit",
        )[0]
        item = layout(
            [kaltsit, mon3tr],
            [
                region("kaltsit-body", "kaltsit", 0.0, 0.5),
                region("mon3tr-body", "mon3tr", 0.5, 0.5),
            ],
        )
        shared, mixer, preview = AnimaRegionalPromptCompilerV2().compile_prompts(
            item,
            "soft painterly scene",
            "Kal'tsit on the left gently pats Mon3tr on the right",
        )
        self.assertEqual(
            shared,
            "soft painterly scene\n"
            "Kal'tsit on the left gently pats Mon3tr on the right",
        )
        self.assertEqual(
            mixer,
            f"{shared}\n"
            "Kal'tsit identity\nstanding on the left, touching Mon3tr\n"
            "Mon3tr identity\nstanding on the right, being touched by Kal'tsit",
        )
        self.assertIn(
            f"Kal'tsit identity\nstanding on the left, touching Mon3tr\n{shared}",
            preview,
        )
        self.assertIn(
            f"Mon3tr identity\nstanding on the right, being touched by Kal'tsit\n{shared}",
            preview,
        )
        self.assertTrue(preview.endswith("[Warnings]\nnone"))

    def test_prompt_compiler_warns_about_explicit_side_conflicts(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice",
            "",
            1.0,
            "",
            "alice",
            "Alice identity",
            "standing on the right",
        )[0]
        item = layout(
            [alice],
            [region("alice-body", "alice", 0.0, 0.4)],
        )
        _, _, preview = AnimaRegionalPromptCompilerV2().compile_prompts(
            item,
            "scene",
            "Alice stands on the right",
        )
        self.assertIn(
            "Alice: Body center is on the left, but prompt text explicitly says right",
            preview,
        )

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

    def test_split_identity_and_pose_prompts_are_optional_and_stable(self):
        node = AnimaRegionalCharacterPromptV2()
        payload = node.build(
            "Alice",
            "",
            1.0,
            "",
            "alice",
            "brown hair, blue eyes, white coat",
            "reaching toward the right",
        )[0]
        self.assertEqual(payload["prompt"], "")
        self.assertEqual(payload["identity_prompt"], "brown hair, blue eyes, white coat")
        self.assertEqual(payload["pose_prompt"], "reaching toward the right")

    def test_prompt_pack_defaults_to_classic_complete_character_branch(self):
        routing_input = AnimaRegionalPromptPackV2.INPUT_TYPES()["optional"]["routing_mode"]
        self.assertEqual(routing_input[0][0], "classic_0_2")
        self.assertIn("global_mix_v1", routing_input[0])
        self.assertEqual(routing_input[1]["default"], "classic_0_2")
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice",
            "",
            1.0,
            "",
            "alice",
            "brown hair, blue eyes, white coat",
            "reaching toward the right",
        )[0]
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(
            clip,
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "shared scene",
            "negative",
            layout_prompt="two people, one on each side",
        )[0]
        shared_scene = "shared scene\ntwo people, one on each side"
        self.assertEqual(
            clip.encoded,
            [
                shared_scene,
                "negative",
                "brown hair, blue eyes, white coat\n"
                "reaching toward the right\n"
                f"{shared_scene}",
            ],
        )
        self.assertEqual(pack["routing_mode"], "classic_0_2")
        self.assertEqual(pack["routing_contract"], "legacy_v2")
        self.assertIn("conditioning", pack["characters"][0])
        self.assertNotIn("identity_conditioning", pack["characters"][0])
        self.assertNotIn("pose_conditioning", pack["characters"][0])
        self.assertIsNone(pack["layout_conditioning"])

    def test_prompt_pack_global_mix_uses_complete_branch_and_actual_base(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice",
            "",
            1.0,
            "",
            "alice",
            "brown hair, blue eyes",
            "reaching toward the right",
        )[0]
        clip = FakeClip()
        base_positive = conditioning(9.0)
        pack = AnimaRegionalPromptPackV2().pack(
            clip,
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "shared scene",
            "negative",
            base_positive=base_positive,
            routing_mode="global_mix_v1",
            global_mix_weight=0.25,
        )[0]
        self.assertEqual(
            clip.encoded,
            [
                "negative",
                "brown hair, blue eyes\nreaching toward the right\nshared scene",
            ],
        )
        self.assertEqual(pack["routing_mode"], "global_mix_v1")
        self.assertEqual(pack["routing_contract"], "global_mix_v1")
        self.assertEqual(pack["global_mix_weight"], 0.25)
        self.assertIs(pack["positive"], base_positive)
        self.assertIsNone(pack["background_conditioning"])
        self.assertIsNone(pack["shared_conditioning"])
        self.assertIn("conditioning", pack["characters"][0])
        self.assertNotIn("identity_conditioning", pack["characters"][0])
        self.assertNotIn("pose_conditioning", pack["characters"][0])

    def test_prompt_pack_rejects_invalid_global_mix_weight(self):
        alice = character("alice", "Alice")
        with self.assertRaisesRegex(ValueError, "global_mix_weight"):
            AnimaRegionalPromptPackV2().pack(
                FakeClip(),
                layout([alice], [region("a", "alice", 0.0, 1.0)]),
                "",
                "",
                routing_mode="global_mix_v1",
                global_mix_weight=2.1,
            )

    def test_prompt_pack_separates_background_layout_identity_and_pose(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice",
            "",
            1.0,
            "",
            "alice",
            "brown hair, blue eyes",
            "reaching toward the right",
        )[0]
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(
            clip,
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "shared scene",
            "negative",
            layout_prompt="two people, one on each side",
            routing_mode="separated_v1_experimental",
        )[0]
        self.assertEqual(
            clip.encoded,
            [
                "shared scene",
                "negative",
                "two people, one on each side",
                "brown hair, blue eyes",
                "reaching toward the right",
            ],
        )
        self.assertEqual(pack["routing_mode"], "separated_v1_experimental")
        self.assertEqual(pack["routing_contract"], "separated_v1")
        self.assertIn("identity_conditioning", pack["characters"][0])
        self.assertIn("pose_conditioning", pack["characters"][0])
        self.assertIsNotNone(pack["layout_conditioning"])

    def test_split_identity_ignores_stale_legacy_prompt_with_warning(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice",
            "old full prompt with duplicated clothes and pose",
            1.0,
            "",
            "alice",
            "brown hair, blue eyes",
            "reaching toward the right",
        )[0]
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(
            clip,
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "background",
            "",
        )[0]
        self.assertNotIn("old full prompt", "\n".join(clip.encoded))
        self.assertTrue(pack["characters"][0]["legacy_prompt_ignored"])
        self.assertEqual(len(pack["warnings"]), 1)

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

    def test_soft_hint_fields_are_canonicalized_and_validated(self):
        alice = character("alice", "Alice")
        result = layout(
            [alice],
            [
                region(
                    "hint",
                    "alice",
                    0.2,
                    0.2,
                    "ownership_hint",
                    hint_blend="soft",
                    strength=0.65,
                )
            ],
        )
        self.assertEqual(result["regions"][0]["hint_blend"], "soft")
        self.assertEqual(result["regions"][0]["strength"], 0.65)
        self.assertEqual(result["regions"][0]["priority"], 0)
        with self.assertRaisesRegex(ValueError, "hint_blend"):
            layout(
                [alice],
                [
                    region(
                        "hint",
                        "alice",
                        0.2,
                        0.2,
                        "ownership_hint",
                        hint_blend="capsule",
                    )
                ],
            )

    def test_region_layer_priority_is_canonicalized_and_validated(self):
        alice = character("alice", "Alice")
        result = layout(
            [alice],
            [region("body", "alice", 0.0, 1.0, priority=3)],
        )
        self.assertEqual(result["regions"][0]["priority"], 3)
        with self.assertRaisesRegex(ValueError, "priority must be an integer"):
            layout(
                [alice],
                [region("body", "alice", 0.0, 1.0, priority=0.5)],
            )
        with self.assertRaisesRegex(ValueError, "priority must be between"):
            layout(
                [alice],
                [region("body", "alice", 0.0, 1.0, priority=11)],
            )
        with self.assertRaisesRegex(ValueError, "strength"):
            layout(
                [alice],
                [
                    region(
                        "hint",
                        "alice",
                        0.2,
                        0.2,
                        "ownership_hint",
                        hint_blend="soft",
                        strength=1.1,
                    )
                ],
            )

    def test_character_is_encoded_once_even_with_two_body_regions(self):
        alice = character("alice", "Alice")
        regions = [
            {"uuid": "r1", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": 0, "y": 0, "width": .5, "height": 1, "feather": 0, "enabled": True},
            {"uuid": "r2", "character_uuid": "alice", "type": "body_region", "geometry": "box", "x": .5, "y": 0, "width": .5, "height": 1, "feather": 0, "enabled": True},
            {"uuid": "hint", "character_uuid": "alice", "type": "ownership_hint", "geometry": "box", "x": .25, "y": 0, "width": .25, "height": 1, "feather": 0, "enabled": True},
        ]
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(clip, layout([alice], regions), "global", "negative")[0]
        self.assertEqual(len(pack["characters"]), 1)
        self.assertEqual(clip.encoded, ["global", "negative", "Alice prompt\nglobal"])
        self.assertTrue(pack["shared_is_final"])
        self.assertTrue(pack["background_is_final"])
        self.assertIs(pack["shared_conditioning"]["raw"], pack["positive"][0][0])

    def test_external_conditioning_remains_final_without_standalone_classic_background(self):
        alice = character("alice", "Alice")
        external_positive, external_negative = conditioning(3.0), conditioning(-1.0)
        clip = FakeClip()
        pack = AnimaRegionalPromptPackV2().pack(clip, layout([alice], []), "global", "negative", external_positive, external_negative)[0]
        model, positive, negative, status = AnimaRegionalApplyV2().apply(FakeModel(), pack, False, "replace")
        self.assertIs(positive, external_positive)
        self.assertIs(negative, external_negative)
        self.assertEqual(status, "disabled; original model returned")
        self.assertEqual(clip.encoded, ["Alice prompt\nglobal"])
        self.assertFalse(pack["shared_is_final"])
        self.assertFalse(pack["background_is_final"])
        self.assertIsNone(pack["shared_conditioning"])
        self.assertIsNone(pack["background_conditioning"])

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

    def test_exclusive_overlap_uses_nearest_body_center_not_character_order(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.6),
            region("b", "bob", 0.4, 0.6),
        ]
        _, forward = build_character_masks(layout([alice, bob], regions), 1, 10)
        _, reordered = build_character_masks(layout([bob, alice], regions), 1, 10)

        forward_by_id = {"alice": forward[0], "bob": forward[1]}
        reordered_by_id = {"bob": reordered[0], "alice": reordered[1]}
        self.assertTrue(torch.equal(forward_by_id["alice"], reordered_by_id["alice"]))
        self.assertTrue(torch.equal(forward_by_id["bob"], reordered_by_id["bob"]))
        self.assertEqual(int(forward_by_id["alice"].sum()), 5)
        self.assertEqual(int(forward_by_id["bob"].sum()), 5)

    def test_higher_layer_priority_wins_only_the_body_overlap(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.6, priority=1),
            region("b", "bob", 0.4, 0.6, priority=0),
        ]
        _, effective = build_character_masks(layout([alice, bob], regions), 1, 10)
        self.assertTrue(torch.all(effective[0, 0, :6] == 1.0))
        self.assertTrue(torch.all(effective[1, 0, 6:] == 1.0))
        self.assertEqual(float(effective[1, 0, 4]), 0.0)

    def test_higher_layer_priority_resolves_overlapping_hints(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("body", "alice", 0.0, 1.0),
            region(
                "alice-hint",
                "alice",
                0.25,
                0.25,
                "ownership_hint",
                priority=0,
            ),
            region(
                "bob-hint",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                priority=2,
            ),
        ]
        _, effective, hint_targets = build_character_mask_components(
            layout([alice, bob], regions),
            1,
            8,
        )
        self.assertEqual(float(effective[0, 0, 2]), 0.0)
        self.assertEqual(float(effective[1, 0, 2]), 1.0)
        self.assertEqual(float(hint_targets[0, 0, 2]), 0.0)
        self.assertEqual(float(hint_targets[1, 0, 2]), 1.0)

    def test_ownership_hint_overrides_body_voronoi_only_inside_hint(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.6),
            region("b", "bob", 0.4, 0.6),
            region("hint", "bob", 0.3, 0.1, "ownership_hint"),
        ]
        _, effective = build_character_masks(layout([alice, bob], regions), 1, 10)
        self.assertEqual(float(effective[1, 0, 3]), 1.0)
        self.assertEqual(float(effective[0, 0, 3]), 0.0)
        self.assertEqual(float(effective[0, 0, 2]), 1.0)

    def test_soft_hint_crossfades_from_current_body_owner(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.5),
            region("b", "bob", 0.5, 0.5),
            region(
                "soft-hint",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        _, effective = build_character_masks(layout([alice, bob], regions), 1, 8)
        self.assertAlmostEqual(float(effective[0, 0, 2]), 0.5)
        self.assertAlmostEqual(float(effective[1, 0, 2]), 0.5)
        self.assertAlmostEqual(float(effective[:, 0, 2].sum()), 1.0)
        self.assertAlmostEqual(float(effective[0, 0, 4]), 0.0)
        self.assertAlmostEqual(float(effective[1, 0, 4]), 1.0)

    def test_hint_target_components_isolate_only_resolved_hint_owner(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.5),
            region("b", "bob", 0.5, 0.5),
            region(
                "soft-hint",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        raw, effective, hint_targets = build_character_mask_components(
            layout([alice, bob], regions),
            1,
            8,
        )
        self.assertEqual(tuple(raw.shape), (2, 1, 8))
        self.assertAlmostEqual(float(effective[0, 0, 2]), 0.5)
        self.assertAlmostEqual(float(effective[1, 0, 2]), 0.5)
        self.assertAlmostEqual(float(hint_targets[0, 0, 2]), 0.0)
        self.assertAlmostEqual(float(hint_targets[1, 0, 2]), 0.5)
        self.assertEqual(float(hint_targets[:, 0, 4].sum()), 0.0)

    def test_soft_hint_outside_body_activates_branch_without_full_coverage(self):
        alice = character("alice", "Alice")
        regions = [
            region("body", "alice", 0.0, 0.5),
            region(
                "soft-hint",
                "alice",
                0.75,
                0.125,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        _, effective = build_character_masks(layout([alice], regions), 1, 8)
        self.assertAlmostEqual(float(effective[0, 0, 6]), 0.5)

    def test_hard_hint_wins_over_overlapping_soft_hint(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("body", "alice", 0.0, 1.0),
            region(
                "soft-hint",
                "alice",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
            region(
                "hard-hint",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="hard",
                strength=1.0,
            ),
        ]
        _, effective = build_character_masks(layout([alice, bob], regions), 1, 8)
        self.assertEqual(float(effective[0, 0, 2]), 0.0)
        self.assertEqual(float(effective[1, 0, 2]), 1.0)

    def test_explicit_hard_hint_matches_legacy_default(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        legacy_regions = [
            region("a", "alice", 0.0, 0.5),
            region("b", "bob", 0.5, 0.5),
            region("hint", "bob", 0.3, 0.1, "ownership_hint"),
        ]
        explicit_regions = [
            region("a", "alice", 0.0, 0.5),
            region("b", "bob", 0.5, 0.5),
            region(
                "hint",
                "bob",
                0.3,
                0.1,
                "ownership_hint",
                hint_blend="hard",
                strength=1.0,
            ),
        ]
        _, legacy = build_character_masks(layout([alice, bob], legacy_regions), 1, 10)
        _, explicit = build_character_masks(layout([alice, bob], explicit_regions), 1, 10)
        self.assertTrue(torch.equal(legacy, explicit))

    def test_early_expansion_keeps_symmetric_voronoi_ownership(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.5),
            region("b", "bob", 0.5, 0.5),
        ]
        _, expanded = build_character_masks(
            layout([alice, bob], regions), 1, 64, body_expand=0.04
        )
        self.assertEqual(int(expanded[0].sum()), 32)
        self.assertEqual(int(expanded[1].sum()), 32)

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
        self.assertIn("route=classic_0_2", status)

    def test_pack_without_route_metadata_remains_classic_compatible(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
        )[0]
        pack.pop("routing_mode")
        pack.pop("routing_contract")
        patched, _, _, status = AnimaRegionalApplyV2().apply(
            FakeModel(),
            pack,
            True,
            "replace",
        )
        state = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ].router.state
        self.assertEqual(state["routing_contract"], "legacy_v2")
        self.assertIn("route=classic_0_2", status)

    def test_classic_routes_uuid_complete_branch_through_ownership_hint(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice", "", 1.0, "", "alice", "Alice identity", "Alice pose"
        )[0]
        bob = AnimaRegionalCharacterPromptV2().build(
            "Bob", "", 1.0, "", "bob", "Bob identity", "Bob pose"
        )[0]
        regions = [
            region("alice-body", "alice", 0.0, 0.5),
            region("bob-body", "bob", 0.5, 0.5),
            region("bob-hand", "bob", 0.25, 0.25, "ownership_hint"),
        ]
        model = FakeModel()
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice, bob], regions),
            "shared scene",
            "",
            conditioning(0.0),
            conditioning(-1.0),
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 10.0
        )
        pack["characters"][1]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )
        patched, _, _, status = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {
                "composition_mode": "early_layout",
                "multi_character_guard": "strong",
                "detail_preserve_mode": "strong",
                "hint_constraint_mode": "strong",
                "character_focus_mode": "adaptive",
                "edge_focus_power": 2.0,
                "identity_anchor_mode": "shared_delta",
            },
        )
        patch = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        state = patch.router.state
        self.assertEqual(state["composition_mode"], "off")
        self.assertEqual(state["multi_character_guard"], "off")
        self.assertEqual(state["detail_preserve_mode"], "off")
        self.assertEqual(state["hint_constraint_mode"], "off")
        self.assertEqual(state["character_focus_mode"], "off")
        self.assertEqual(state["identity_anchor_mode"], "off")
        self.assertFalse(state["has_identity_conditioning"])
        self.assertIn("post_0_2_enhancements=ignored", status)

        state["input_shape"] = (1, 4, 4, 8)
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.zeros((1, 2, 4)),
            transformer_options={"cond_or_uncond": [0]},
        )
        expected = torch.tensor(
            [[10.0, 20.0, 20.0, 20.0, 10.0, 20.0, 20.0, 20.0]]
        ).unsqueeze(-1).expand_as(output)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [0.0, 10.0, 20.0],
        )

    def test_classic_soft_hint_crossfades_complete_branches_and_preserves_unconditional(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice", "", 1.0, "", "alice", "Alice identity", "Alice pose"
        )[0]
        bob = AnimaRegionalCharacterPromptV2().build(
            "Bob", "", 1.0, "", "bob", "Bob identity", "Bob pose"
        )[0]
        regions = [
            region("alice-body", "alice", 0.0, 0.5),
            region("bob-body", "bob", 0.5, 0.5),
            region(
                "bob-hand-soft",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice, bob], regions),
            "shared scene",
            "",
            conditioning(0.0),
            conditioning(-1.0),
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 10.0
        )
        pack["characters"][1]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )
        model = FakeModel()
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
        )
        patch = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        patch.router.state["input_shape"] = (2, 4, 4, 8)
        base_context = torch.tensor([0.0, 100.0]).view(2, 1, 1).expand(
            2, 2, 4
        )
        output = patch(
            torch.zeros((2, 8, 4)),
            base_context,
            transformer_options={"cond_or_uncond": [0, 1]},
        )
        expected_conditional = torch.tensor(
            [10.0, 15.0, 20.0, 20.0, 10.0, 15.0, 20.0, 20.0]
        ).view(8, 1).expand(8, 4)
        self.assertTrue(torch.equal(output[0], expected_conditional))
        self.assertTrue(torch.equal(output[1], torch.full_like(output[1], 100.0)))
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [0.0, 10.0, 20.0],
        )

    def test_global_mix_blends_actual_base_complete_branches_and_soft_hint(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice", "", 1.0, "", "alice", "Alice identity", "Alice pose"
        )[0]
        bob = AnimaRegionalCharacterPromptV2().build(
            "Bob", "", 1.0, "", "bob", "Bob identity", "Bob pose"
        )[0]
        regions = [
            region("alice-body", "alice", 0.0, 0.5),
            region("bob-body", "bob", 0.5, 0.5),
            region(
                "bob-hand-soft",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice, bob], regions),
            "shared scene",
            "",
            conditioning(100.0),
            conditioning(-1.0),
            routing_mode="global_mix_v1",
            global_mix_weight=0.25,
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 10.0
        )
        pack["characters"][1]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )
        model = FakeModel()
        patched, _, _, status = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "base_preserve",
        )
        patch = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        patch.router.state["input_shape"] = (2, 4, 4, 8)
        base_context = torch.tensor([100.0, 200.0]).view(2, 1, 1).expand(
            2, 2, 4
        )
        output = patch(
            torch.zeros((2, 8, 4)),
            base_context,
            transformer_options={"cond_or_uncond": [0, 1]},
        )
        expected_conditional = torch.tensor(
            [28.0, 32.0, 36.0, 36.0, 28.0, 32.0, 36.0, 36.0]
        ).view(8, 1).expand(8, 4)
        self.assertTrue(torch.allclose(output[0], expected_conditional))
        self.assertTrue(torch.equal(output[1], torch.full_like(output[1], 200.0)))
        self.assertIn("route=global_mix_v1", status)
        self.assertIn("global/base_weight=0.25", status)
        self.assertIn("nominal_mix=20% base + 80% character @ strength=1", status)
        self.assertIn("blend=bounded_absolute", status)
        self.assertEqual(patch.router.state["global_mix_weight"], 0.25)
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 10.0, 20.0],
        )

    def test_global_mix_keeps_uncovered_tokens_on_actual_base(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout(
                [alice, bob],
                [
                    region("alice-body", "alice", 0.0, 0.25),
                    region("bob-body", "bob", 0.75, 0.25),
                ],
            ),
            "",
            "",
            conditioning(100.0),
            routing_mode="global_mix_v1",
            global_mix_weight=0.25,
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 10.0
        )
        pack["characters"][1]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            FakeModel(), pack, True, "replace"
        )
        patch = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        patch.router.state["input_shape"] = (1, 4, 2, 8)
        output = patch(
            torch.zeros((1, 4, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        expected = torch.tensor([28.0, 100.0, 100.0, 36.0]).view(
            1, 4, 1
        ).expand_as(output)
        self.assertTrue(torch.allclose(output, expected))

    def test_global_mix_detail_preserve_fades_character_mix_only_late(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("alice-body", "alice", 0.0, 1.0)]),
            "",
            "",
            conditioning(100.0),
            routing_mode="global_mix_v1",
            global_mix_weight=0.25,
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )
        patched, _, _, status = AnimaRegionalApplyV2().apply(
            FakeModel(),
            pack,
            True,
            "replace",
            {
                "detail_preserve_mode": "soft",
                "detail_preserve_start": 0.65,
                "detail_preserve_amount": 0.25,
            },
        )
        patch = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        state = patch.router.state
        self.assertEqual(state["detail_preserve_mode"], "soft")
        self.assertIn("detail=soft (start=0.65, amount=0.25)", status)
        state["input_shape"] = (1, 4, 4, 8)

        # Without sampling metadata, behavior remains the exact 0.4.4 mix.
        output_without_sigma = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(
            torch.allclose(output_without_sigma, torch.full_like(output_without_sigma, 36.0))
        )

        state["detail_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.8
        output_early = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(
            torch.allclose(output_early, torch.full_like(output_early, 36.0))
        )

        state["current_sigma"] = 0.0
        output_late = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(
            torch.allclose(output_late, torch.full_like(output_late, 52.0))
        )

    def test_adaptive_character_focus_tracks_local_complete_branch_saliency(self):
        state = {
            "character_focus_mode": "adaptive",
            "input_shape": (1, 4, 14, 14),
            "patch_spatial": 2,
            "patch_temporal": 1,
        }
        x = torch.zeros((1, 49, 4))
        base = torch.zeros_like(x)
        values = torch.full((7, 7), 0.25)
        values[2:5, 2:5] = 2.0
        branch = values.reshape(1, 49, 1).expand_as(x)
        ownership = torch.ones((1, 49, 1))

        focus = _adaptive_character_focus(
            state, x, branch, base, ownership
        ).reshape(7, 7)
        self.assertGreater(float(focus[3, 3]), 0.9)
        self.assertLess(float(focus[0, 0]), 0.1)
        self.assertTrue(torch.all((focus >= 0.0) & (focus <= 1.0)))

        uniform = _adaptive_character_focus(
            state, x, torch.ones_like(x), base, ownership
        )
        self.assertTrue(torch.allclose(uniform, torch.full_like(uniform, 0.5)))
        state["character_focus_mode"] = "off"
        self.assertTrue(
            torch.equal(
                _adaptive_character_focus(state, x, branch, base, ownership),
                torch.zeros_like(ownership),
            )
        )

    def test_global_mix_adaptive_focus_and_hint_use_the_stronger_late_hold(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout(
                [alice],
                [
                    region("alice-body", "alice", 0.0, 1.0),
                    region(
                        "alice-hand",
                        "alice",
                        0.25,
                        0.25,
                        "ownership_hint",
                    ),
                ],
            ),
            "",
            "",
            conditioning(100.0),
            routing_mode="global_mix_v1",
            global_mix_weight=0.25,
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )
        model = FakeModel()
        patched, _, _, status = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {
                "detail_preserve_mode": "soft",
                "detail_preserve_start": 0.7,
                "detail_preserve_amount": 0.3,
                "hint_constraint_mode": "soft",
                "character_focus_mode": "adaptive",
            },
        )
        patch = patched.object_patches[
            "diffusion_model.blocks.0.cross_attn.forward"
        ]
        state = patch.router.state
        state["input_shape"] = (1, 4, 4, 8)
        state["detail_sigma_range"] = (0.0, 1.0)
        state["sampling_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.8
        output_early = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(
            torch.equal(output_early, torch.full_like(output_early, 36.0))
        )
        state["current_sigma"] = 0.0
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )

        expected = torch.tensor(
            [50.4, 45.6, 50.4, 50.4, 50.4, 45.6, 50.4, 50.4]
        ).view(1, 8, 1).expand_as(output)
        self.assertTrue(torch.allclose(output, expected, atol=1e-4))
        self.assertIn("character_focus=adaptive", status)
        self.assertNotIn("split-route enhancements=ignored", status)
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 20.0, 100.0, 20.0],
        )

    def test_global_mix_soft_hint_hold_strengthens_only_late_hint_targets(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("alice-body", "alice", 0.0, 0.5),
            region("bob-body", "bob", 0.5, 0.5),
            region(
                "bob-hand-soft",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice, bob], regions),
            "",
            "",
            conditioning(100.0),
            routing_mode="global_mix_v1",
            global_mix_weight=0.25,
        )[0]
        pack["characters"][0]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 10.0
        )
        pack["characters"][1]["conditioning"]["raw"] = torch.full(
            (1, 2, 4), 20.0
        )

        outputs = {}
        for hint_mode in ("off", "soft"):
            patched, _, _, status = AnimaRegionalApplyV2().apply(
                FakeModel(),
                pack,
                True,
                "replace",
                {
                    "detail_preserve_mode": "soft",
                    "detail_preserve_start": 0.7,
                    "detail_preserve_amount": 0.3,
                    "hint_constraint_mode": hint_mode,
                },
            )
            patch = patched.object_patches[
                "diffusion_model.blocks.0.cross_attn.forward"
            ]
            state = patch.router.state
            state["input_shape"] = (1, 4, 4, 8)
            state["detail_sigma_range"] = (0.0, 1.0)
            state["sampling_sigma_range"] = (0.0, 1.0)
            state["current_sigma"] = 0.0
            outputs[hint_mode] = patch(
                torch.zeros((1, 8, 4)),
                torch.full((1, 2, 4), 100.0),
                transformer_options={"cond_or_uncond": [0]},
            )
            if hint_mode == "soft":
                self.assertIn("hint_late_hold=soft", status)

        expected_off = torch.tensor(
            [49.6, 52.4, 55.2, 55.2, 49.6, 52.4, 55.2, 55.2]
        ).view(1, 8, 1).expand_as(outputs["off"])
        expected_soft = torch.tensor(
            [49.6, 47.6, 55.2, 55.2, 49.6, 47.6, 55.2, 55.2]
        ).view(1, 8, 1).expand_as(outputs["soft"])
        self.assertTrue(torch.allclose(outputs["off"], expected_off))
        self.assertTrue(torch.allclose(outputs["soft"], expected_soft))

    def test_global_mix_zero_base_weight_matches_classic_at_unit_strength(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("alice-body", "alice", 0.0, 0.5),
            region("bob-body", "bob", 0.5, 0.5),
            region(
                "soft",
                "bob",
                0.25,
                0.25,
                "ownership_hint",
                hint_blend="soft",
                strength=0.5,
            ),
        ]
        outputs = []
        for routing_mode in ("classic_0_2", "global_mix_v1"):
            pack = AnimaRegionalPromptPackV2().pack(
                FakeClip(),
                layout([alice, bob], regions),
                "",
                "",
                conditioning(100.0),
                routing_mode=routing_mode,
                global_mix_weight=0.0,
            )[0]
            pack["characters"][0]["conditioning"]["raw"] = torch.full(
                (1, 2, 4), 10.0
            )
            pack["characters"][1]["conditioning"]["raw"] = torch.full(
                (1, 2, 4), 20.0
            )
            patched, _, _, _ = AnimaRegionalApplyV2().apply(
                FakeModel(), pack, True, "replace"
            )
            patch = patched.object_patches[
                "diffusion_model.blocks.0.cross_attn.forward"
            ]
            patch.router.state["input_shape"] = (1, 4, 4, 8)
            outputs.append(
                patch(
                    torch.zeros((1, 8, 4)),
                    torch.full((1, 2, 4), 100.0),
                    transformer_options={"cond_or_uncond": [0]},
                )
            )
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_apply_filters_inactive_branches_and_honors_global_strength(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice, bob], [region("a", "alice", 0.0, 1.0)]),
            "",
            "",
            routing_mode="separated_v1_experimental",
        )[0]
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            FakeModel(), pack, True, "replace", {
                "global_strength": 0.25,
                "boundary_falloff": 2,
                "multi_character_guard": "soft",
                "detail_preserve_mode": "soft",
                "detail_preserve_start": 0.65,
                "detail_preserve_amount": 0.5,
                "hint_constraint_mode": "strong",
                "character_focus_mode": "adaptive",
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
        self.assertEqual(state["hint_constraint_mode"], "off")
        self.assertEqual(state["character_focus_mode"], "off")
        self.assertEqual(state["edge_focus_power"], 1.5)
        state["input_shape"] = (1, 4, 4, 8)
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.zeros((1, 2, 4)),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 0.25)))

    def test_multi_character_guard_keeps_attention_output_three_dimensional(self):
        alice, bob = character("alice", "Alice"), character("bob", "Bob")
        regions = [
            region("a", "alice", 0.0, 0.75),
            region("b", "bob", 0.25, 0.75),
        ]
        for guard in ("soft", "strong"):
            with self.subTest(guard=guard):
                pack = AnimaRegionalPromptPackV2().pack(
                    FakeClip(),
                    layout([alice, bob], regions),
                    "background",
                    "",
                    conditioning(100.0),
                    routing_mode="separated_v1_experimental",
                )[0]
                pack["background_conditioning"]["raw"] = torch.full(
                    (1, 2, 4), 10.0
                )
                pack["characters"][0]["identity_conditioning"]["raw"] = torch.full(
                    (1, 2, 4), 13.0
                )
                pack["characters"][1]["identity_conditioning"]["raw"] = torch.full(
                    (1, 2, 4), 17.0
                )
                patched, _, _, _ = AnimaRegionalApplyV2().apply(
                    FakeModel(),
                    pack,
                    True,
                    "replace",
                    {"multi_character_guard": guard},
                )
                patch = patched.object_patches[
                    "diffusion_model.blocks.0.cross_attn.forward"
                ]
                state = patch.router.state
                state["input_shape"] = (2, 4, 4, 8)
                context = torch.tensor([100.0, 200.0]).view(2, 1, 1).expand(
                    2, 2, 4
                )
                output = patch(
                    torch.zeros((2, 8, 4)),
                    context,
                    transformer_options={"cond_or_uncond": [0, 1]},
                )

                self.assertEqual(tuple(output.shape), (2, 8, 4))
                self.assertEqual(tuple(state["last_overlap_mask"].shape), (2, 8, 1))
                self.assertTrue(torch.any(state["last_overlap_mask"][0] > 0.0))
                self.assertTrue(torch.all(output[1] == 200.0))

    def test_shared_delta_uses_character_minus_shared_scene(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
            routing_mode="separated_v1_experimental",
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
            routing_mode="separated_v1_experimental",
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
            routing_mode="separated_v1_experimental",
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

    def test_separated_pack_uses_one_background_relative_identity_residual(self):
        alice = character("alice", "Alice")
        model = FakeModel()
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
            routing_mode="separated_v1_experimental",
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
        self.assertTrue(torch.allclose(output, torch.full_like(output, 102.4)))
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 10.0, 13.0],
        )

    def test_separated_pack_routes_group_layout_once_over_body_union(self):
        alice = character("alice", "Alice")
        model = FakeModel()
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "background",
            "",
            conditioning(100.0),
            layout_prompt="one person centered",
            routing_mode="separated_v1_experimental",
        )[0]
        pack["background_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["layout_conditioning"]["raw"] = torch.full((1, 2, 4), 20.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 13.0)
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model, pack, True, "replace", {"layout_strength": 1.0}
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        patch.router.state["input_shape"] = (1, 4, 4, 8)
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertTrue(torch.allclose(output, torch.full_like(output, 113.0)))
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 10.0, 20.0, 13.0],
        )

    def test_shared_delta_broadcasts_context_and_preserves_runtime_dtype(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
            routing_mode="separated_v1_experimental",
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
            routing_mode="separated_v1_experimental",
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

    def test_late_identity_detail_uses_shared_delta_and_core_only_inside_body(self):
        alice = AnimaRegionalCharacterPromptV2().build(
            "Alice",
            "",
            1.0,
            "",
            "alice",
            "brown hair, blue eyes, white coat",
            "reaching toward the right",
        )[0]
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
            routing_mode="separated_v1_experimental",
        )[0]
        pack["shared_conditioning"]["raw"] = torch.full((1, 2, 4), 10.0)
        pack["characters"][0]["conditioning"]["raw"] = torch.full((1, 2, 4), 13.0)
        pack["characters"][0]["identity_conditioning"]["raw"] = torch.full(
            (1, 2, 4), 15.0
        )
        model = FakeModel()
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {
                "identity_detail_mode": "late",
                "identity_detail_start": 0.5,
                "identity_detail_strength": 1.0,
                "identity_core_strength": 1.5,
                "identity_core_radius": 0.55,
            },
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        state = patch.router.state
        state["input_shape"] = (1, 4, 4, 8)
        state["sampling_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.0
        output = patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 10.0, 1.0, 15.0],
        )
        self.assertTrue(torch.all(output >= 13.0))
        self.assertGreater(float(output.max()), 13.0)
        self.assertTrue(state["identity_context_cache"])
        self.assertEqual(state["identity_core_cache"], {})

    def test_late_identity_detail_separated_pack_adds_no_duplicate_branch(self):
        alice = character("alice", "Alice")
        pack = AnimaRegionalPromptPackV2().pack(
            FakeClip(),
            layout([alice], [region("a", "alice", 0.0, 1.0)]),
            "global",
            "",
            conditioning(100.0),
            routing_mode="separated_v1_experimental",
        )[0]
        model = FakeModel()
        patched, _, _, _ = AnimaRegionalApplyV2().apply(
            model,
            pack,
            True,
            "replace",
            {
                "identity_detail_mode": "late",
                "identity_detail_start": 0.5,
                "identity_detail_strength": 1.0,
            },
        )
        patch = patched.object_patches["diffusion_model.blocks.0.cross_attn.forward"]
        state = patch.router.state
        state["input_shape"] = (1, 4, 4, 8)
        state["sampling_sigma_range"] = (0.0, 1.0)
        state["current_sigma"] = 0.0
        patch(
            torch.zeros((1, 8, 4)),
            torch.full((1, 2, 4), 100.0),
            transformer_options={"cond_or_uncond": [0]},
        )
        self.assertEqual(
            model.diffusion_model.blocks[0].cross_attn.context_means,
            [100.0, 1.0, 1.0],
        )
        self.assertTrue(state["identity_context_cache"])


if __name__ == "__main__":
    unittest.main()
