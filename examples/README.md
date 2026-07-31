# Example workflow

`Anima-Regional-0.3.3-Mixer-Amiya-Kaltsit-identity-anchor.json` demonstrates:

- two independent Character Prompt nodes;
- overlapping Body regions and local Ownership Hints for crossed hands;
- one Shared Scene Prompt feeding both Mixer and Regional Prompt Pack;
- Post-Adapter Mixer before Regional Apply;
- `base_preserve` routing;
- the 0.3.3 `shared_delta` identity anchor.

The workflow expects these node families:

- this repository;
- Anima Artist Mixer;
- the Anima sampler node providing `AnimaFlowCorrectiveSampler`.

Adjust the CLIP, UNET, and VAE filenames to match your installation.

The example deliberately keeps framing and action in the Shared Scene Prompt,
while character identity, ears, hair, eyes, clothing, and accessories remain in
their individual Character Prompt nodes.
