# ComfyUI Anima Regional Prompt 0.3.4

Training-free character-level regional prompting for Anima. The plugin keeps
character prompts in separate attention branches and routes each branch with an
editable spatial ownership layout. It does not modify `comfy/ldm` or the Anima
model weights.

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```text
git clone https://github.com/An1X3R/ComfyUI-Anima-Regional.git
```

Restart ComfyUI after installation or update.

An importable Mixer + Regional workflow is included in `examples/`. Its model
filenames are examples and may need to be changed for your installation.

## 0.3.3 identity anchor

This release adds an opt-in `shared_delta` identity anchor for character
features and clothing. With the ordinary route, a regional branch is compared
directly with the final base branch. When the base comes from Post-Adapter
Mixer, that difference can unintentionally weaken late identity details.

`shared_delta` instead measures:

```text
character prompt plus shared scene - shared scene alone
```

That cleaner character residual is then added to the real final base, including
the Mixer's style result. With no external Mixer positive, the shared scene is
already the base and is reused without another attention call. With an external
positive, one shared attention branch is added per patched layer and denoising
call, independent of the number of characters.

The recommended first A/B settings are:

```text
identity_anchor_mode = shared_delta
identity_anchor_strength = 1.0
identity_late_floor = 0.8
detail_preserve_mode = soft
edge_focus_power = 1.0
```

Keep camera, composition, action, and framing terms such as `upper body` in the
shared scene prompt. Keep each character name, unique hair/eyes/ears, clothing,
and accessories in that character's own prompt. Do not place character identity
or clothing for all subjects in the shared prompt: shared terms are deliberately
subtracted by the identity anchor.

## 0.3.4 identity core (development)

Character Prompt now has two optional fields for crowded scenes:

- `identity prompt (stable traits)`: name, hair, eyes, ears or horns, clothing,
  and accessories that should remain attached to that character.
- `pose / interaction prompt`: position, facing, gesture, leaning, touching, and
  other action or composition terms.

The original `prompt` field remains compatible with 0.3.3 workflows. If either
new field is filled, the main regional branch combines the split fields and the
old field; the old field can be left empty or used for extra character-local
details. If both new fields are empty, the old path is unchanged and no
identity-only branch is evaluated.

The optional `identity_detail_mode=late` adds a second, clean identity residual
only during late denoising. It compares `identity prompt + shared scene` with
the shared scene, so it does not rewrite the whole pose branch. The residual is
strongest in the interior of Body boxes and is multiplied by the existing
ownership mask; it never expands ownership and it ignores Ownership Hints for
the core boost. This is intended for four-plus characters standing close
together, not as a hard segmentation guarantee.

The late branch costs one shared attention call plus one identity call per
character only after `identity_detail_start`. Legacy prompts or
`identity_detail_mode=off` add zero extra attention calls.

## V2 workflow

Use the nodes in `Anima/Regional`:

1. Create one `Anima Regional - Character Prompt` per character. For the
   crowded-scene route, put stable traits in `identity prompt` and actions in
   `pose / interaction prompt`; leave the legacy prompt empty unless needed.
2. Connect them to `Anima Regional - Layout`. Draw Body regions and optional
   Ownership Hints directly on its canvas.
3. Enter the shared scene description once in `Anima Regional - Shared Scene Prompt`.
   Connect its STRING output to both the Artist Pack `base_prompt` and Regional
   Prompt Pack `global_prompt` inputs. A STRING output can fan out to both nodes.
4. Connect Layout and CLIP to `Anima Regional - Prompt Pack`. Use its
   `negative prompt` field, or connect external negative conditioning.
5. Connect the pack and MODEL to `Anima Regional - Apply`.
6. Use `Anima Regional - Inspect Masks` when precise black/white mask outputs
   are needed.

The Prompt Pack labels intentionally distinguish the two positive paths:
`shared scene context` is text used to encode regional branches, while
`external base positive (Mixer output)` is the already encoded final positive
conditioning from Post-Adapter Mixer. Connecting the latter makes it the one
authoritative final positive output and avoids a duplicate conditioning branch.

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
- Ownership Hint: in `exclusive`, a small box locally assigns its pixels to one
  character, including pixels outside that character's Body region. It never
  affects pixels outside its own box. V2 intentionally ignores Hints in
  `normalized` mode.

Hints are intended for crossed hands, legs, hair, clothing, or an upper body
entering another character's broad region. Keep them local; a large hard Hint
can suppress useful interaction context.

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
Anchor-Q path for the base and every character branch, then performs the outer
K/V ownership routing.

Full Artist Cross-Attn Mixer, unknown attention patches, reverse-order patching,
and applying Regional twice are rejected with an explicit error. Do not chain
the legacy full Cross-Attn Mixer with Regional.

## Advanced options

`Anima Regional Advanced Options` is separate from the main workflow. V2 uses:

- `global_strength`
- `start_block` / `end_block`
- `start_percent` / `end_percent`
- optional `boundary_falloff`
- `composition_mode`: `off` (the 0.2.x behavior) or `early_layout`
- `composition_strength`: early-phase layout emphasis, from 0 to 2
- `composition_expand`: temporary Body-region expansion, from 0 to 0.15
- `composition_end_percent`: when the temporary composition help fades out
- `multi_character_guard`: `off`, `soft`, or `strong` protection for 3+ active
  character branches
- `detail_preserve_mode`: fade regional rewriting during late denoising
- `detail_preserve_start` / `detail_preserve_amount`: control when and how far
  the late fade proceeds
- `edge_focus_power`: reduce influence only at already-soft mask edges
- `identity_anchor_mode`: `off` or the new `shared_delta` identity residual
- `identity_anchor_strength`: blend from the old base delta at 0 to the clean
  shared-scene-subtracted delta at 1
- `identity_late_floor`: minimum late identity influence while
  `detail_preserve_mode` is fading the regional route
- `identity_detail_mode`: `off` or `late`; enables the optional late identity
  residual when split identity prompts exist
- `identity_detail_start`: normalized denoising progress at which that residual
  begins to fade in
- `identity_detail_strength`: overall strength of the late identity residual
- `identity_core_strength`: extra multiplier at the center of Body boxes
- `identity_core_radius`: fraction of each Body box treated as its central core

Layout owns `overlap_mode`, and Apply owns `blend_mode`; their duplicate values
inside the compatibility Options payload do not override V2. Legacy strict
self-attention settings are also ignored by V2.

`boundary_falloff=0` is the unchanged baseline. In exclusive mode, a nonzero
value attenuates the winning branch near internal character ownership
boundaries without activating the losing character. It has no effect in
normalized mode.

`early_layout` is a training-free composition aid. During the first part of
sampling it briefly widens Body regions and increases the regional branch
delta, then returns to ordinary routing. Ownership Hints are never widened.
This can make broad placement more stable, but it is not a hard pose or
instance-segmentation constraint. Start with `composition_strength=1.0`,
`composition_expand=0.04`, and `composition_end_percent=0.55`.

V2 Character Prompt strength accepts 0 to 4. Values up to 2 retain the old
scaling. Values above 2 use a residual-norm cap relative to the base attention
output, so the extra range is deliberately conservative rather than an
unbounded multiplier. If a seed becomes unstable, return to 1.0 to 2.0.

For scenes with three or more active characters, `multi_character_guard` borrows
the Mixer's base-anchored blending idea without pooling character embeddings:
each character keeps its own branch, but `replace` no longer replaces the base
output wholesale, and the combined residual receives a per-token budget.
`soft` is the recommended starting point; `strong` is for difficult four-plus
character layouts. The guard is opt-in, so old workflows with `off` are
unchanged. `base_preserve` already has the base anchor and therefore only uses
the residual budget when a guard is enabled.

`detail_preserve_mode` addresses late-stage blur and anatomy damage by gradually
returning fine-detail control to the base branch. Without identity anchoring it
fades the whole regional residual, including some clothing and facial detail.
With `shared_delta`, `identity_late_floor` prevents that clean identity residual
from fading too far. Start with `soft`, `start=0.65`, `amount=0.5`, and
`identity_late_floor=0.8`; use `strong` only when a four-plus character image
still breaks down. If the checkpoint wrapper does not expose sampling sigma,
the feature falls back to the unchanged route instead of guessing.

`identity_anchor_strength=0` is an exact numerical A/B baseline and skips the
extra shared attention call. A value of `1` uses only the character-minus-shared
residual. Intermediate values blend the two routes and introduce the late
identity floor gradually.

For the new identity detail route, start with `late`, `start=0.6`,
`strength=0.65`, `core_strength=1.5`, and `core_radius=0.55`. If the model
wrapper does not expose sampling sigma, the route stays off rather than running
at every step. The added branch is deliberately capped relative to the base
attention output; increasing the control can improve clothing retention, but
it cannot turn the feature into a hard mask or prevent all self-attention
communication.

`edge_focus_power=1.0` is unchanged behavior. Values around `1.3` to `1.6`
attenuate only fractional mask weights, keeping fully owned interiors at full
strength while reducing contamination and blur near feathers or boundaries.
It does not shrink hard binary Body regions.

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
