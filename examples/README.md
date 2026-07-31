# Example workflow

`Anima-Regional-0.4.0-Mixer-Amiya-Kaltsit-separated.json` demonstrates:

- two independent Character Prompt nodes;
- overlapping Body regions and local Ownership Hints for crossed hands;
- one background/style Shared Scene Prompt feeding both Mixer and Regional Prompt Pack;
- a separate Prompt Pack `layout_prompt` for count, framing and crossed hands;
- Post-Adapter Mixer before Regional Apply;
- `base_preserve` routing;
- the `separated_v1` background/layout/pose/identity route;
- the V2-only Routing Options node without legacy strict controls.

The workflow expects these node families:

- this repository;
- Anima Artist Mixer;
- the Anima sampler node providing `AnimaFlowCorrectiveSampler`.

Adjust the CLIP, UNET, and VAE filenames to match your installation.

The example keeps only background/style in the Shared Scene Prompt. Group
framing and interaction are in `layout_prompt`; character identity and local
action are split between the two Character Prompt fields.

The older `Anima-Regional-0.3.3-Mixer-Amiya-Kaltsit-identity-anchor.json` remains
in the repository as a compatibility reference. Its legacy prompt fields still
load, but the new example is the recommended starting point.
