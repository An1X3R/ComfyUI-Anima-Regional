# ComfyUI Anima Regional Prompt 0.4.0 (routing redesign)

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

## 0.4.0 routing contract

The V2 Prompt Pack now keeps four roles separate:

```text
final Mixer/base branch
+ box-localized (layout/presence - background)
+ owner-localized (pose - layout)
+ owner-localized (identity - background)
```

This remains training-free and attention-based, but multi-person count and
interaction text are no longer copied into every character identity branch.
Identity is injected once with one shared residual budget. The old
`shared_delta` fields remain accepted for saved workflows; new
`separated_v1` packs select the split route automatically.

The practical prompt rule is:

- `global_prompt`: background, lighting, camera and style only.
- `layout_prompt`: number of people, framing, positions and interaction.
- `identity_prompt`: one character's name, hair, eyes, ears/horns, clothing and
  accessories.
- `pose_prompt`: that character's facing, gesture, reaching, leaning or contact.

The final positive remains the external Post-Adapter Mixer output when it is
connected. Regional evaluates the standalone background once only when that
external base must be aligned, then adds localized residuals to the real base.

The former `shared_delta` description was:

```text
character prompt plus shared scene - shared scene alone
```

That compatibility route is still available for old packs. For new packs use:

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
3. Enter background/style once in `Anima Regional - Shared Scene Prompt`.
   Connect its STRING output to both the Artist Pack base prompt and Regional
   Prompt Pack `global_prompt` inputs. Put group count and interaction in the
   Prompt Pack `layout_prompt`, not in the background string.
4. Connect Layout and CLIP to `Anima Regional - Prompt Pack`. Use its
   `negative prompt` field, or connect external negative conditioning.
5. Connect the pack and MODEL to `Anima Regional - Apply`.
6. Use `Anima Regional - Inspect Masks` when precise black/white mask outputs
   are needed.

The Prompt Pack labels distinguish background, layout and external positive
paths. `external base positive (Mixer output)` is the already encoded final
positive conditioning from Post-Adapter Mixer. Connecting it makes it the one
authoritative final positive output; regional branches are added as attention
residuals instead of concatenated conditioning lists.

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
- Ownership Hint is an explicit local override, including pixels outside that
  character's Body region. It never affects pixels outside its own box. V2
  intentionally ignores Hints in `normalized` mode.

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
Anchor-Q path for base, background, layout, pose and identity branches, then
performs the outer K/V ownership routing.

Full Artist Cross-Attn Mixer, unknown attention patches, reverse-order patching,
and applying Regional twice are rejected with an explicit error. Do not chain
the legacy full Cross-Attn Mixer with Regional.

## Advanced options

`Anima Regional - V2 Routing Options` is separate from the main workflow. It
contains only controls consumed by the V2 router:

- `global_strength`
- `start_block` / `end_block`
- `start_percent` / `end_percent`
- optional `boundary_falloff`
- `composition_mode`: `off` (the 0.2.x behavior) or `early_layout`
- `composition_strength`: early-phase layout emphasis, from 0 to 2
- `composition_expand`: temporary Body-region expansion, from 0 to 0.15
- `composition_end_percent`: when the temporary composition help fades out
- `layout_strength`: strength of the shared layout/presence residual
- `multi_character_guard`: an overlap-only residual cap; it never scales a
  character down merely because more characters are present
- `detail_preserve_mode`: fade regional rewriting during late denoising
- `detail_preserve_start` / `detail_preserve_amount`: control when and how far
  the late fade proceeds
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

`early_layout` is a training-free composition aid. During the first part of
sampling it briefly widens only the layout/presence route and increases its
residual, then returns to ordinary routing. Identity ownership continues to
use the original Body boxes. Ownership Hints are never widened.
This can make broad placement more stable, but it is not a hard pose or
instance-segmentation constraint. Start with `composition_strength=1.0`,
`composition_expand=0.04`, and `composition_end_percent=0.55`.

V2 Character Prompt strength accepts 0 to 4. Values up to 2 retain the old
scaling. Values above 2 use a residual-norm cap relative to the base attention
output, so the extra range is deliberately conservative rather than an
unbounded multiplier. If a seed becomes unstable, return to 1.0 to 2.0.

`multi_character_guard` is local to actual overlap. It does not multiply a
character branch by `sqrt(2/N)` or `2/N`; a clean, non-overlapping token keeps
its full identity strength even when four characters are active. `soft` adds a
moderate cap in overlap strips, while `strong` is for visibly conflicting
contact areas.

`detail_preserve_mode` addresses late-stage blur and anatomy damage by gradually
returning layout and pose control to the base branch. The isolated identity
residual has its own `identity_late_floor`, so clothing and facial traits are
not protected by accidentally keeping the whole pose branch alive. Start with
`soft`, `start=0.65`, `amount=0.5`, and `identity_late_floor=0.8`. If the
checkpoint wrapper does not expose sampling sigma, the feature falls back to
unchanged behavior instead of guessing.

`identity_anchor_mode` and the old late-identity controls are retained only for
legacy packs. New `separated_v1` packs already use the background-relative
identity residual once; a second late identity branch is intentionally not part
of the V2 UI.

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
