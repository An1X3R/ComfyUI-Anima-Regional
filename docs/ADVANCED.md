# ComfyUI Anima Regional Prompt 0.4.7 (adaptive character focus)

Training-free character-level regional prompting for Anima. The plugin supports
an explicit complete-character `classic_0_2` route, a bounded
`global_mix_v1` route, and a retained experimental `separated_v1_experimental`
route, all driven by an editable spatial ownership layout. It does not modify
`comfy/ldm` or the Anima model weights.

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```text
git clone https://github.com/An1X3R/ComfyUI-Anima-Regional.git
```

Restart ComfyUI after installation or update.

An importable Mixer + Regional workflow is included in `examples/`. Its model
filenames are examples and may need to be changed for your installation.
The existing `*-separated.json` example is an experimental comparison workflow,
not the recommended classic recovery baseline.

## 0.4.7 Adaptive character focus

`global_mix_v1` can now retain more of a complete character branch at spatial
tokens where that branch differs meaningfully from the actual Mixer/base
output. This is a semantic, layout-local alternative to protecting the fixed
center of a Body box: it can follow a face, hair, clothing feature, or prompted
gesture without assuming that the subject is upright or centered.

`character_focus_mode` is appended after `hint_constraint_mode` in V2 Options:

- `off`: exact 0.4.6 behavior;
- `adaptive`: during late detail preservation, restore at most half of the
  released character influence according to normalized local branch saliency.

The saliency calculation reuses the complete character and Mixer/base outputs
that Global Mix already evaluates. It adds no prompt branch or attention call,
uses a small spatial average to avoid token noise, and remains inactive before
the detail-preserve schedule begins. Character focus and Ownership Hint hold
take the stronger local correction rather than adding together, so effective
ownership remains non-negative and bounded by the original mask.

Recommended first A/B values:

```text
global_mix_weight = 0.20
detail_preserve_mode = soft
detail_preserve_start = 0.75
detail_preserve_amount = 0.20
hint_constraint_mode = soft
character_focus_mode = adaptive
```

The feature is deliberately conservative: a large character/base difference
can still represent a malformed feature, so this release does not expose an
aggressive focus strength.

## 0.4.6 Ownership Hint late hold

`global_mix_v1` can now keep resolved Ownership Hint targets stronger than the
surrounding Body region during late detail preservation. This addresses the
tradeoff where returning broad regional coverage to the actual Mixer/base
improves fabric and accessory detail but can also soften a hand, forearm, or
other cross-region limb inside a Hint.

The new `hint_constraint_mode` is appended to the V2 Options node so existing
widget values keep their positions:

- `off`: numerically preserves the 0.4.5 late-detail behavior;
- `soft`: restores half of the Hint correction released by detail preserve;
- `strong`: retains the resolved Hint target through the final denoising stage.

The implementation does not add another prompt branch or attention call. It
keeps the Global Mix weights non-negative and bounded, and changes only tokens
covered by a resolved Ownership Hint. Recommended first test:

```text
detail_preserve_mode = soft
detail_preserve_start = 0.70
detail_preserve_amount = 0.30
hint_constraint_mode = soft
```

This improves local character ownership and limb clarity; it is not a pose or
skeleton estimator, so exact joint geometry still requires OpenPose/Depth when
the base model cannot resolve the requested interaction.

## 0.4.5 Global Mix late detail preserve

`global_mix_v1` can now use the existing late `detail_preserve_mode`. The
bounded character/base mix remains unchanged through early and middle
denoising, then optionally fades part of the regional contribution so the
actual Mixer/base branch can finish fine line work, fabric texture, small
accessories, and other high-frequency detail.

The feature is default-off. Saved workflows with `detail_preserve_mode=off`
remain numerically identical to 0.4.4. If sampling sigma metadata is not
available, it also falls back to the unchanged result instead of guessing.
For a slight detail-softening problem, start with:

```text
detail_preserve_mode = soft
detail_preserve_start = 0.70
detail_preserve_amount = 0.30
```

This is intentionally a late schedule rather than a higher global base weight:
identity, layout, and Ownership Hints keep their original strength while the
image structure is being established, and only the final detail stage shifts
toward the base.

## 0.4.4 Global Mix workflow UX

This release deliberately leaves the visually validated `global_mix_v1`
attention formula unchanged. It makes the existing mode easier to configure:

- the Global Mix balance label now translates the raw base weight into an
  approximate base/character percentage at character strength `1.0`;
- the Layout toolbar can mirror the selected region or the complete layout
  horizontally without changing character UUID ownership;
- for exactly two connected characters, a selected Ownership Hint can create a
  mirrored reciprocal Hint for the other character in one action;
- copied or mirrored Hints retain `strength`, `feather`, `priority`, enabled
  state, and hard/soft blend mode.

These operations only edit serialized layout geometry and ownership. They do
not add prompt branches, change routing weights, or alter old workflows that do
not use the new buttons.

## 0.4.3 Global Mix and layer priority

`global_mix_v1` keeps the actual incoming positive conditioning—including the
Post-Adapter Mixer's final context—as a bounded global/base contribution inside
owned regions. Every character still uses one complete UUID branch:

```text
identity_prompt + pose_prompt + shared scene
```

The Prompt Pack adds one user-facing control, `global_mix_weight`. Start at
`0.25`. With character strength `1.0`, a fully covered token is approximately
80% complete character branch and 20% actual base. `0` approaches classic
replacement, while `1` gives approximately equal character/base weight.

V2 Layout also restores the early layer-like `priority` control. Higher values
win only where regions of the same ownership stage overlap; equal values keep
the automatic nearest-center decision. The default is `0`, so existing layouts
remain unchanged. Hard Ownership Hints still override Body ownership, and
priority only resolves Body-vs-Body, Hard-Hint-vs-Hard-Hint, or
Soft-Hint-vs-Soft-Hint intersections.

## 0.4.2 classic recovery

This version restores an explicit `classic_0_2` route with one complete
UUID-owned character branch (`identity + pose + shared scene`), adds a Prompt
Compiler so Regional and Artist Mixer consume synchronized text, and adds an
optional Soft Ownership Hint. Code tests and the real ComfyUI interface smoke
check are covered here; image quality and limb behavior still require
controlled same-seed user-side rendering.

## 0.4.1 hotfix

`multi_character_guard=soft` and `strong` now keep the overlap mask in the
normal `[batch, token, 1]` shape. Version 0.4.0 accidentally retained an extra
leading dimension when the guard was active, which changed cross-attention
output from three dimensions to four and caused Anima's final `einops`
rearrange to fail. The Mixer warning was only the outer wrapper reporting this
Regional runtime error.

## Routing contracts

The V2 Prompt Pack exposes an explicit `routing_mode` selector. The default is
`classic_0_2`; `global_mix_v1` is the bounded quality/context A/B route;
`separated_v1_experimental` remains available only for controlled comparison.

### `classic_0_2` (default baseline)

Each character UUID is encoded once as one complete branch:

```text
identity_prompt + pose_prompt + global_prompt + layout_prompt
```

The runtime then uses the 0.2-style absolute attention replacement:

```text
base + Σ mask_i * strength_i * (complete_character_i - base)
```

Body regions and that character's Ownership Hints always select the same
complete branch. With an external Mixer positive, the Mixer output remains the
authoritative base; classic does not calculate a separate background residual.

For a faithful baseline, start with character strength `1.0` to `2.0`,
`global_strength=1.0`, `blend_mode=replace`, and the post-0.2 enhancement
controls disabled. The current classic implementation intentionally keeps the
current ownership masks; it does not claim to restore the historical 0.2
Voronoi/order rules.

### `global_mix_v1`

Global Mix evaluates the same complete character branches as classic, but
combines absolute attention outputs instead of adding an unbounded residual.
For an exclusive owner with spatial coverage `mask`:

```text
regional_share = character_strength / (global_mix_weight + character_strength)
output = base + mask * regional_share * (complete_character - base)
```

The implementation generalizes this formula to Soft Hints and normalized Body
overlaps while keeping the sum bounded. Coverage remains outside the
normalization, so feathered edges remain feathered and uncovered tokens return
the actual base. CFG/unconditional rows also remain on the base path.

Global Mix deliberately ignores `base_preserve` projection and the later split-
route composition/identity enhancement controls. It optionally consumes the
late detail-preserve schedule described above. The Apply status reports
`blend=bounded_absolute`. It continues to use the same Q-only Anchor callable
for the actual base and every complete character branch.

Recommended first A/B values:

- `global_mix_weight=0.25`: character-forward; best first test after classic.
- `global_mix_weight=0.5`: more Mixer/global context, softer ownership.
- `global_mix_weight=1.0`: equal base/character weight at strength `1.0`.

Global Mix aims to preserve scene coherence and partial interactions; it is not
a pose, skeleton, OpenPose, or depth constraint.

### `separated_v1_experimental`

This route retains the 0.4.1 split residual experiment:

```text
final Mixer/base branch
+ box-localized (layout/presence - background)
+ owner-localized (pose - layout)
+ owner-localized (identity - background)
```

Use it only as an A/B control. It keeps the separate background, layout, pose,
and identity attention evaluations and is not the recommended recovery path.

The practical prompt fields are:

- `global_prompt`: shared scene, background, lighting, camera, style, and other
  terms that should describe the whole image.
- `layout_prompt`: shared group count, framing, positions, and interaction.
- `identity_prompt`: one character's name, hair, eyes, ears/horns, clothing and
  accessories.
- `pose_prompt`: that character's facing, gesture, reaching, leaning or contact.
- `global_mix_weight`: actual base weight used only by `global_mix_v1`.

In classic mode, the shared scene and layout text are merged into every complete
character branch. In separated mode, they retain their separate residual roles.

The former `shared_delta` description was:

```text
character prompt plus shared scene - shared scene alone
```

That compatibility route is still available for old packs. It is ignored by the
classic baseline so it cannot silently change the 0.2 comparison.

For an experimental separated-route comparison, start with:

```text
blend_mode = base_preserve
layout_strength = 1.0
global_strength = 1.0
multi_character_guard = off
identity_late_floor = 0.8
detail_preserve_mode = soft
edge_focus_power = 1.0
```

## Prompt migration

Character Prompt now has two optional fields for crowded scenes:

- `identity prompt (stable traits)`: name, hair, eyes, ears or horns, clothing,
  and accessories that should remain attached to that character.
- `pose / interaction prompt`: position, facing, gesture, leaning, touching, and
  other action or composition terms.

The original `prompt` field remains accepted. If `identity_prompt` is filled,
the old field is ignored and Apply reports a migration warning. If only
`pose_prompt` is filled, the old field is used as the stable identity fallback.
This avoids silently adding the same full character description twice.

## V2 workflow

Use the nodes in `Anima/Regional`:

1. Create one `Anima Regional - Character Prompt` per character. For the
   crowded-scene route, put stable traits in `identity prompt` and actions in
   `pose / interaction prompt`; leave the legacy prompt empty unless needed.
2. Connect them to `Anima Regional - Layout`. Draw Body regions and optional
   Ownership Hints directly on its canvas.
3. For Mixer workflows, prefer `Anima Regional - Prompt Compiler`. Connect
   Layout once, enter scene/style text in `scene_prompt` and the relationship or
   interaction once in `interaction_prompt`. Connect its
   `regional_shared_prompt` to Prompt Pack `global_prompt` and its
   `mixer_full_context` to the Artist Pack base prompt. The compiler also emits
   a read-only preview and warns when Body-side geometry contradicts explicit
   left/right text.
4. Without the compiler, `Anima Regional - Shared Scene Prompt` may still fan
   out to both inputs, but the user must keep the Mixer context and Regional
   scene text synchronized manually. Put optional group count and interaction
   in the shared/interaction text; classic mode merges it into every complete
   character branch.
5. Connect Layout and CLIP to `Anima Regional - Prompt Pack`. Leave
   `routing_mode=classic_0_2` for the recovery baseline. Use
   `global_mix_v1` with `global/base weight=0.25` for the first bounded A/B.
   Use `separated_v1_experimental` only when an explicit comparison is intended.
   Use its `negative prompt` field, or connect external negative conditioning.
6. Connect the pack and MODEL to `Anima Regional - Apply`.
7. Use `Anima Regional - Inspect Masks` when precise black/white mask outputs
   are needed.

The Prompt Pack labels distinguish shared scene, layout, routing mode and
external positive paths. `external base positive (Mixer output)` is the already
encoded final positive conditioning from Post-Adapter Mixer. Connecting it
makes it the one authoritative final positive output; classic regional branches
replace the selected ownership areas relative to that base instead of adding a
separate background branch.

Character sockets on Layout stay compact: connected sockets plus one empty
socket are shown, up to eight characters. Regions bind to stable character
UUIDs, not socket positions, so socket reorder does not reassign ownership.

The Layout canvas supports selecting, dragging, resizing, copying, deleting,
numeric fine adjustment, undo/redo transactions, and optional 1/32 grid snap.
Disconnected regions are retained as editor-only orphans and return when the
same character is reconnected.

The editor canvas now preserves the configured image aspect ratio inside its
stable node viewport. Portrait layouts appear portrait, landscape layouts appear
landscape, and the normalized region coordinates remain unchanged.

## Ownership

Each character can have multiple Body regions. They are max-unioned into one
runtime branch mask, so adding another box does not encode the prompt again.

- `exclusive`: every active image token has at most one character owner.
- `normalized`: overlapping Body masks are normalized and blended.
- In `exclusive`, overlapping Body boxes use the nearest original Body center
  (a Voronoi decision), so character socket order no longer steals the whole
  overlap strip.
- Each selected region has `layer priority` from `-10` to `10`. Higher priority
  wins only in an overlap; equal priority preserves the automatic Voronoi rule.
  Priority is ignored by `normalized` Body blending.
- Ownership Hint is an explicit local override, including pixels outside that
  character's Body region. It never affects pixels outside its own box. V2
  intentionally ignores Hints in `normalized` mode.

Hints are intended for crossed hands, legs, hair, clothing, or an upper body
entering another character's broad region. Keep them local; a large hard Hint
can suppress useful interaction context. A Hint has two optional controls:

- `hint_blend=hard` (default): historical full owner replacement.
- `hint_blend=soft`: where another Body/Hard-Hint owner already exists,
  interpolates from that owner to the Hint owner's complete UUID branch.
  Outside all existing ownership coverage, the remaining weight stays on the
  actual base branch. `strength` controls the interpolation amount and defaults
  to `1.0`.

The current editor still stores Hint geometry as an axis-aligned box. A future
Limb Link/Capsule editor is intentionally not part of this compatibility step.

## Post-Adapter Mixer compatibility

The supported model order is:

```text
MODEL -> Anima Artist Post-Adapter Mixer -> Anima Regional - Apply -> sampler
```

For positive conditioning compatibility, connect the Mixer's final positive to
`external base positive (Mixer output)` on Prompt Pack. Connect its MODEL output
to Apply. Prompt Pack then returns that one authoritative positive instead of
concatenating another independent positive list. Keep the Artist Pack's
`artist_chain` limited to artist/style tokens; character identity belongs in
the Regional Character Prompt nodes.

Q-only Anchor is supported when its attention callable advertises the public
`_anima_adapter_anchor_q_forward_patch = True` marker. Regional calls the same
Anchor-Q path for the base and every complete character branch in
`classic_0_2` and `global_mix_v1`. The experimental separated route additionally calls its
background, layout, pose and identity branches before performing the outer K/V
ownership routing.

Full Artist Cross-Attn Mixer, unknown attention patches, reverse-order patching,
and applying Regional twice are rejected with an explicit error. Do not chain
the legacy full Cross-Attn Mixer with Regional.

## Advanced options

`Anima Regional - V2 Routing Options` is separate from the main workflow. It
contains only controls consumed by the V2 router:

The `classic_0_2` baseline uses only the common strength, block/sigma range,
boundary falloff, and blend controls. `global_mix_v1` uses strength,
block/sigma range, boundary falloff, its Prompt Pack base-weight control, and
the optional late detail-preserve schedule; its bounded absolute formula
ignores the Apply blend selector. Composition, multi-character guard, edge
focus, and identity-anchor controls are ignored by both complete-branch routes
and remain available for the experimental separated route.

- `global_strength`
- `start_block` / `end_block`
- `start_percent` / `end_percent`
- optional `boundary_falloff`
- `composition_mode`: experimental early-layout assistance; ignored by classic
- `composition_strength`: early-phase layout emphasis, from 0 to 2
- `composition_expand`: temporary Body-region expansion, from 0 to 0.15
- `composition_end_percent`: when the temporary composition help fades out
- `layout_strength`: strength of the shared layout/presence residual
- `multi_character_guard`: an overlap-only residual cap; it never scales a
  character down merely because more characters are present
- `detail_preserve_mode`: fade regional rewriting during late denoising
- `detail_preserve_start` / `detail_preserve_amount`: control when and how far
  the late fade proceeds
- `hint_constraint_mode`: Global Mix-only late hold for resolved Ownership Hint
  targets; appended after the legacy fields to preserve saved widget positions
- `character_focus_mode`: Global Mix-only adaptive late hold for locally salient
  complete-character output; appended after `hint_constraint_mode`
- `edge_focus_power`: reduce influence only at already-soft mask edges
- `identity_late_floor`: minimum late influence for the isolated identity
  residual, never for the combined pose/layout branch

Layout owns `overlap_mode`, and Apply owns `blend_mode`. Legacy Options nodes
remain available for old workflows, but their self-attention and branch-chunk
controls are compatibility-only and are not part of V2.

`boundary_falloff=0` is the unchanged baseline. In exclusive mode, a nonzero
value attenuates the winning branch near internal character ownership
boundaries without activating the losing character. It has no effect in
normalized mode.

In `separated_v1_experimental`, `early_layout` is a training-free composition
aid. During the first part of sampling it briefly widens only the
layout/presence route and increases its residual, then returns to ordinary
routing. Identity ownership continues to use the original Body boxes.
Ownership Hints are never widened. Classic mode ignores this control.
This can make broad placement more stable, but it is not a hard pose or
instance-segmentation constraint. Start with `composition_strength=1.0`,
`composition_expand=0.04`, and `composition_end_percent=0.55`.

V2 Character Prompt strength accepts 0 to 4 for saved-workflow compatibility.
For `classic_0_2`, use `1.0` to `2.0`; values above 2 are outside the historical
0.2 baseline and are not recommended for A/B. The experimental separated route
uses a residual-norm cap for its extended range. A route-specific classic
high-strength policy remains intentionally undecided until visual evidence is
available.

In the experimental separated route, `multi_character_guard` is local to actual
overlap. It does not multiply a character branch by `sqrt(2/N)` or `2/N`; a
clean, non-overlapping token keeps its full identity strength even when four
characters are active. `soft` adds a moderate cap in overlap strips, while
`strong` is for visibly conflicting contact areas.

In `global_mix_v1`, `detail_preserve_mode` scales only the final difference
between the bounded regional mix and the actual base. It does not create a new
prompt branch, modify masks, or change early identity/Hint strength. For mild
detail recovery, start with `soft`, `start=0.70`, and `amount=0.30`.

When `hint_constraint_mode` is enabled, the same bounded mix retains additional
target-character weight only inside resolved Ownership Hints. `soft` is the
recommended limb A/B setting; `strong` can resist late ownership drift more
aggressively but may preserve a malformed limb when the base model's anatomy is
already wrong.

When `character_focus_mode=adaptive`, Global Mix compares each complete
character output with the actual base at the already-owned spatial tokens,
normalizes that difference within the character region, and smooths the result
on the latent grid. During late detail preservation, low-saliency areas keep
the ordinary fade while high-saliency areas regain at most half of the released
character contribution. Hint hold and character focus use the larger local
correction rather than accumulating.

In the experimental separated route, `detail_preserve_mode` addresses
late-stage blur and anatomy damage by gradually returning layout and pose
control to the base branch. The isolated identity residual has its own
`identity_late_floor`, so clothing and facial traits are not protected by
accidentally keeping the whole pose branch alive. Start with `soft`,
`start=0.65`, `amount=0.5`, and `identity_late_floor=0.8`. If the checkpoint
wrapper does not expose sampling sigma, the feature falls back to unchanged
behavior instead of guessing.

`identity_anchor_mode` and the old late-identity controls remain accepted for
saved workflows. Classic ignores them to preserve the recovery baseline. The
experimental separated route already uses the background-relative identity
residual once; a second late identity branch is intentionally not part of the
V2 UI.

In the experimental separated route, `edge_focus_power=1.0` is unchanged
behavior. Values around `1.3` to `1.6` attenuate only fractional mask weights,
keeping fully owned interiors at full strength while reducing contamination and
blur near feathers or boundaries. It does not shrink hard binary Body regions.

## Inspect outputs

`Anima Regional - Inspect Masks` returns:

- `ownership_preview`: colored character ownership.
- `region_masks`: effective per-character MASK batch.
- `box_masks_image`: black/white IMAGE batch for every enabled raw Body or Hint
  box separately.
- `effective_masks_image`: black/white effective per-character IMAGE batch.

## Legacy and limits

The 0.1.2 nodes remain available under `Anima/Regional/Legacy`; old class IDs,
widget order, and execution paths are preserved. `strict_experimental` remains
a legacy research control because real-image tests produced hard partitions and
duplicate subjects.

Exclusive routing substantially reduces direct prompt contamination, but it
cannot guarantee zero leakage. Self-attention, residual features, VAE decoding,
and the base model's learned composition still communicate globally. Ownership
Hints improve local identity binding; they do not provide instance segmentation
or guarantee anatomically correct limbs at every seed.

## Contributing

Pull requests and reproducible real-model A/B reports are welcome. Read
`CONTRIBUTING.md` before changing node interfaces, attention compatibility, or
saved-workflow widget order.
