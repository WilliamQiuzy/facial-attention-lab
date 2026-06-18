"""Per-baseline adapters into our project's evaluation pipeline.

Strategy (locked 2026-05-26):

| Baseline | What we do | Output target |
|---|---|---|
| Gaber ASI | train HB head on our data once labels arrive | HB I..VI |
| Oo MLP-Mixer | train HB head on cached frozen embeddings; optionally full fine-tune | HB I..VI |
| Baumann ResNet-18 | fine-tune released weights on our data | HB I..VI |
| Shagdar GCNMid | keep released weights, evaluate native binary task on our data | binary (stroke/nonstroke) |

Shagdar is the only baseline that is NOT converted to HB. The other three
become HB classifiers via these adapters; Shagdar provides a binary
comparison baseline.

See `HB_ADAPTERS.md` for full design + retraining-decision rationale.
"""
from .oo_mixer_for_hb import OoMixerForHB, OoMixerForHBConfig
from .gaber_asi_for_hb import GaberASIForHB, GaberASIForHBConfig
from .baumann_for_hb import BaumannForHB, BaumannForHBConfig
from .shagdar_binary_eval import ShagdarBinaryEval, ShagdarBinaryEvalConfig

__all__ = [
    "OoMixerForHB", "OoMixerForHBConfig",
    "GaberASIForHB", "GaberASIForHBConfig",
    "BaumannForHB", "BaumannForHBConfig",
    "ShagdarBinaryEval", "ShagdarBinaryEvalConfig",
]
