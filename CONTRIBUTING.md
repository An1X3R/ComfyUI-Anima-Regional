# Contributing

Contributions are welcome. Please keep changes focused and preserve the
training-free design of the plugin.

## Development layout

- `anima_regional/v2/`: current character-level routing backend.
- `anima_regional/`: legacy backend, shared options, and compatibility helpers.
- `web/js/anima_regional_layout_v2.js`: interactive region editor.
- `tests/`: unit, Mixer/Q-only compatibility, and ComfyUI interface smoke tests.
- `examples/`: importable ComfyUI workflows.

## Compatibility rules

- Do not modify ComfyUI's `comfy/ldm` files or model weights.
- Preserve existing node class IDs and saved-workflow widget order.
- Add new Advanced Options fields at the end with compatibility-safe defaults.
- Post-Adapter Mixer is the supported Mixer path.
- Q-only Anchor must remain the inner attention callable for the base, shared,
  and character branches.
- Full Cross-Attn Mixer and unknown attention patches must remain rejected.

## Before opening a pull request

Run:

```text
python -m unittest discover -s tests -v
python -m compileall -q .
node --check web/js/anima_regional_layout_v2.js
```

When testing inside a ComfyUI installation, also run:

```text
python tests/smoke_comfy.py
```

Real-model image A/B results are valuable, but they do not replace the automated
checks. Include the seed, resolution, sampler, blend mode, Advanced Options,
character count, and whether Post-Adapter Mixer/Q-only Anchor were enabled.

Please avoid committing model files, generated images, local logs, caches, or
machine-specific absolute paths.
