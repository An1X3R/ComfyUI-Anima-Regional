# Example workflows

`Anima-Regional-0.4.7-Global-Mix-Adaptive-Focus.json` is the recommended
starting point. It demonstrates:

- two editable Character Prompt nodes with separate identity and pose fields;
- Body regions plus small, soft UUID-bound Ownership Hints;
- one Prompt Compiler feeding synchronized Regional and Mixer context;
- the Artist Mixer before Regional Apply;
- `global_mix_v1`, soft late detail preservation, and adaptive character focus;
- VAE decode and output nodes kept after the model sampler.

The workflow uses the standard `CLIPLoader`, `UNETLoader`, `VAELoader`,
`AnimaArtistPack`, `AnimaArtistAdapterMixer`, and
`AnimaFlowCorrectiveSampler` node families. Change the CLIP, UNET, VAE, and
artist settings for your installation. The example does not include model
weights or personal LoRAs.

`Anima-Regional-0.4.0-Mixer-Amiya-Kaltsit-separated.json` is retained as an
experimental compatibility reference. `Anima-Regional-0.3.3-Mixer-Amiya-`
`Kaltsit-identity-anchor.json` is the legacy reference.
