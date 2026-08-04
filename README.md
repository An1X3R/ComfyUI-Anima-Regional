# ComfyUI Anima Regional

Training-free, character-level regional prompt routing for Anima in ComfyUI.
The plugin keeps character identity, pose, shared scene context, and spatial
ownership in one editable layout, then routes the complete character branches
through the model without changing Anima weights or `comfy/ldm`.

## Features

- `global_mix_v1`: recommended mode for multi-character scenes. It combines
  regional character branches with the actual Mixer/base output and includes
  conservative late detail preservation.
- `classic_0_2`: compatibility baseline with complete character branches.
- Separate `identity_prompt` and `pose_prompt` fields. They remain separately
  editable, but are merged with the shared scene at execution time.
- UUID-bound Body regions and Ownership Hints for approximate character
  ownership of crossed arms, hands, clothing, and other interactions.
- `AnimaRegionalPromptCompilerV2` for sending the same scene and interaction
  context to both the regional pack and the Artist Mixer.
- Optional adaptive character focus for preserving identity-bearing local
  features during late detail restoration.

This is attention routing, not geometric control. For exact limbs or camera
placement, use OpenPose, Depth, or another structural control method as well.

## Installation

Clone into the ComfyUI custom-nodes directory and restart ComfyUI:

```text
git clone https://github.com/An1X3R/ComfyUI-Anima-Regional.git ComfyUI-Anima-Regional
```

The plugin requires Python 3.10+ and the Anima Artist Mixer and sampler nodes
used by your workflow. No model weights are included.

## Quick use

1. Add one `AnimaRegionalCharacterPromptV2` node per character. Edit identity
   and pose independently and keep each character UUID stable.
2. Connect the character nodes to `AnimaRegionalLayoutV2`. Use Body regions for
   broad ownership and small, feathered Ownership Hints for interactions.
3. Connect the layout and scene/interaction text to
   `AnimaRegionalPromptCompilerV2`. Connect its `mixer_full_context` output to
   the Artist Mixer base prompt and its `regional_shared_prompt` output to the
   Regional Prompt Pack global prompt.
4. Feed the Mixer base conditioning and the layout into
   `AnimaRegionalPromptPackV2`. Select `global_mix_v1` and connect the pack to
   `AnimaRegionalApplyV2`.
5. Connect the Mixer model to Apply, then Apply to the Anima sampler. Keep the
   VAE decode and preview/save nodes after the sampler; do not connect an IMAGE
   to a layout or prompt node.

## Recommended starting values

```text
routing_mode          = global_mix_v1
global_mix_weight      = 0.20
detail_preserve_mode   = soft
detail_preserve_start  = 0.75
detail_preserve_amount = 0.20
hint_constraint_mode   = soft
character_focus_mode   = adaptive
```

Start with two characters, broad non-feathered Body boxes, and only the hints
needed for the interaction. If identity becomes too weak, raise regional
strength gradually before increasing the mix weight.

## Example workflow

Import [`examples/Anima-Regional-0.4.7-Global-Mix-Adaptive-Focus.json`](examples/Anima-Regional-0.4.7-Global-Mix-Adaptive-Focus.json).
It demonstrates two characters, a shared compiler context, Global Mix,
adaptive focus, and soft Ownership Hints. Change the CLIP, UNET, and VAE
filenames to match your installation before running it.

The older workflows in `examples/` are retained as compatibility references.
The detailed routing and migration notes are in
[`docs/ADVANCED.md`](docs/ADVANCED.md).

## License

MIT. See [`LICENSE`](LICENSE).
