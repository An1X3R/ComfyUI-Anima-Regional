"""Training-free regional prompt routing for Anima."""

from .nodes import (
    AnimaRegionalApply,
    AnimaRegionalCharacterBox,
    AnimaRegionalCharacterMask,
    AnimaRegionalOptions,
    AnimaRegionalPreview,
)
from .v2 import (
    AnimaRegionalApplyV2,
    AnimaRegionalCharacterPromptV2,
    AnimaRegionalInspectV2,
    AnimaRegionalLayoutV2,
    AnimaRegionalPromptPackV2,
    AnimaRegionalOptionsV2,
    AnimaRegionalSharedPromptV2,
)

NODE_CLASS_MAPPINGS = {
    "AnimaRegionalCharacterBox": AnimaRegionalCharacterBox,
    "AnimaRegionalCharacterMask": AnimaRegionalCharacterMask,
    "AnimaRegionalOptions": AnimaRegionalOptions,
    "AnimaRegionalApply": AnimaRegionalApply,
    "AnimaRegionalPreview": AnimaRegionalPreview,
    "AnimaRegionalCharacterPromptV2": AnimaRegionalCharacterPromptV2,
    "AnimaRegionalLayoutV2": AnimaRegionalLayoutV2,
    "AnimaRegionalPromptPackV2": AnimaRegionalPromptPackV2,
    "AnimaRegionalOptionsV2": AnimaRegionalOptionsV2,
    "AnimaRegionalSharedPromptV2": AnimaRegionalSharedPromptV2,
    "AnimaRegionalApplyV2": AnimaRegionalApplyV2,
    "AnimaRegionalInspectV2": AnimaRegionalInspectV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaRegionalCharacterBox": "[Legacy] Anima Character Region (Box)",
    "AnimaRegionalCharacterMask": "[Legacy] Anima Character Region (Mask)",
    "AnimaRegionalOptions": "Anima Regional Advanced Options",
    "AnimaRegionalApply": "[Legacy] Anima Regional Prompt (Encode + Apply)",
    "AnimaRegionalPreview": "[Legacy] Anima Regional Preview",
    "AnimaRegionalCharacterPromptV2": "Anima Regional - Character Prompt",
    "AnimaRegionalLayoutV2": "Anima Regional - Layout",
    "AnimaRegionalPromptPackV2": "Anima Regional - Prompt Pack",
    "AnimaRegionalOptionsV2": "Anima Regional - V2 Routing Options",
    "AnimaRegionalSharedPromptV2": "Anima Regional - Shared Scene Prompt",
    "AnimaRegionalApplyV2": "Anima Regional - Apply",
    "AnimaRegionalInspectV2": "Anima Regional - Inspect Masks",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
