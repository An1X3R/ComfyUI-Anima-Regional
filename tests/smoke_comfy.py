import ast
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_ROOT = Path(sys.executable).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", PORTABLE_ROOT / "ComfyUI"))

for source_path in sorted(PLUGIN_ROOT.rglob("*.py")):
    ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Attention, Block, MiniTrainDIT
from anima_regional import NODE_CLASS_MAPPINGS
from anima_regional.runtime import RegionalCrossAttention, StrictRegionalSelfAttention

plugin_spec = importlib.util.spec_from_file_location(
    "comfyui_anima_regional_plugin",
    PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
assert plugin_spec is not None and plugin_spec.loader is not None
plugin_module = importlib.util.module_from_spec(plugin_spec)
sys.modules[plugin_spec.name] = plugin_module
plugin_spec.loader.exec_module(plugin_module)


assert issubclass(Anima, MiniTrainDIT)
assert callable(getattr(Anima, "preprocess_text_embeds", None))
assert "transformer_options" in inspect.signature(Attention.forward).parameters
assert "transformer_options" in inspect.signature(Block.forward).parameters
assert "AnimaRegionalApply" in NODE_CLASS_MAPPINGS
assert "AnimaRegionalApply" in plugin_module.NODE_CLASS_MAPPINGS
for node_id in (
    "AnimaRegionalCharacterPromptV2",
    "AnimaRegionalLayoutV2",
    "AnimaRegionalPromptPackV2",
    "AnimaRegionalSharedPromptV2",
    "AnimaRegionalApplyV2",
    "AnimaRegionalInspectV2",
):
    assert node_id in NODE_CLASS_MAPPINGS
    assert node_id in plugin_module.NODE_CLASS_MAPPINGS
assert plugin_module.WEB_DIRECTORY == "./web"

torch = __import__("torch")
torch.manual_seed(7)
attention = Attention(
    query_dim=8,
    context_dim=8,
    n_heads=2,
    head_dim=4,
    operations=torch.nn,
).eval()
x = torch.randn(1, 5, 8)
contexts = [torch.randn(1, 6, 8), torch.randn(1, 6, 8)]
reference = [attention(x, context) for context in contexts]
wrapper = RegionalCrossAttention(attention.forward, attention, {}, 0)
optimized = wrapper._q_reuse_chunk(x, contexts, {})
for expected, actual in zip(reference, optimized):
    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)

validation_state = {
    "branch_chunk_size": 2,
    "q_reuse_validated": None,
    "q_reuse_failed": False,
    "q_reuse_logged": False,
}
validation_wrapper = RegionalCrossAttention(
    attention.forward, attention, validation_state, 0
)
validation_candidate = validation_wrapper._q_reuse_chunk(x, contexts, {})
validation_max_diff = max(
    (expected - actual).abs().max().item()
    for expected, actual in zip(reference, validation_candidate)
)
print(f"Automatic Q reuse max abs diff: {validation_max_diff:.9g}")
validated = validation_wrapper._branch_outputs(x, contexts, None, {})
assert validation_state["q_reuse_validated"] is True
assert validation_state["q_reuse_failed"] is False
for expected, actual in zip(reference, validated):
    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)

fallback_state = {
    "branch_chunk_size": 2,
    "q_reuse_validated": None,
    "q_reuse_failed": False,
    "q_reuse_logged": False,
}
fallback_wrapper = RegionalCrossAttention(
    attention.forward, attention, fallback_state, 0
)
fallback_wrapper._q_reuse_chunk = lambda query, chunk, options: [
    torch.zeros_like(query) for _ in chunk
]
runtime_logger = logging.getLogger("anima_regional.runtime")
previous_disabled = runtime_logger.disabled
runtime_logger.disabled = True
try:
    fallback = fallback_wrapper._branch_outputs(x, contexts, None, {})
finally:
    runtime_logger.disabled = previous_disabled
assert fallback_state["q_reuse_validated"] is False
assert fallback_state["q_reuse_failed"] is True
for expected, actual in zip(reference, fallback):
    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)

self_attention = Attention(
    query_dim=8,
    context_dim=None,
    n_heads=2,
    head_dim=4,
    operations=torch.nn,
).eval()
self_state = {
    "self_attention_mode": "strict_experimental",
    "self_attn_max_tokens": 64,
    "regions": [
        {
            "geometry": "box",
            "center_x": 0.5,
            "center_y": 0.5,
            "width": 1.0,
            "height": 1.0,
            "feather": 0.0,
            "strength": 0.0,
            "priority": 0,
        }
    ],
    "overlap_mode": "exclusive",
    "input_shape": (1, 4, 2, 10),
    "patch_spatial": 2,
    "patch_temporal": 1,
    "mask_cache": {},
    "sigma_range": None,
    "cond_or_uncond": [0],
}
self_reference = self_attention(x)
self_wrapper = StrictRegionalSelfAttention(
    self_attention.forward, self_attention, self_state, 0
)
self_masked = self_wrapper.forward(x, transformer_options={"cond_or_uncond": [0]})
assert torch.allclose(self_reference, self_masked, rtol=1e-5, atol=1e-5)

print(f"Parsed plugin Python files: {len(list(PLUGIN_ROOT.rglob('*.py')))}")
print(f"Anima base: {Anima.__mro__[1].__name__}")
print(f"Attention.forward: {inspect.signature(Attention.forward)}")
print(f"Registered nodes: {len(NODE_CLASS_MAPPINGS)}")
print("V2 node registration and WEB_DIRECTORY: OK")
print("ComfyUI root-package import: OK")
print("Real Attention Q reuse equivalence: OK")
print("Automatic Q reuse validation and fallback: OK")
print("Real self-attention all-allowed equivalence: OK")
print("ComfyUI interface smoke check: OK")
