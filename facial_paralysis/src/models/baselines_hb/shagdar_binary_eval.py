"""Shagdar 2024 GNN — binary evaluator on our data (no HB conversion).

Project direction (2026-05-26): we keep Shagdar's released `.pt` weights
verbatim and evaluate the model **on its native binary stroke/nonstroke
task**, not on House-Brackmann. Rationale:

  - Shagdar already released weights, so head-surgery + retraining is a
    larger lift than just using what was published.
  - The GCNMid backbone was trained on binary supervision (Toronto NeuroFace
    stroke vs nonstroke); whether its 256-d features transfer to HB ordinal
    severity is uncertain and not worth speculating on without trying.
  - A binary baseline is still useful to compare against our HB system:
    derive a binary collapse of HB (e.g. {I} = nonstroke, {II..VI} = stroke)
    and compute agreement against Shagdar's binary predictions.

This module wraps the GCNMid inference path so it can be called
programmatically. It does NOT load weights at construction — pass a
checkpoint path to `predict(...)` so we can ensemble across the 4 clean
splits (or 3 pose splits) the paper ships.

Returns per-graph probability of `stroke` (class 1) ∈ [0, 1]. Argmax to get
binary class.

Usage:
    from src.models.baselines_hb import ShagdarBinaryEval
    evaluator = ShagdarBinaryEval()
    p_stroke = evaluator.predict_graph(x, edge_index, batch,
                                       weights="src/baselines/shagdar_gnn/model_weights/clean_split_0.pt")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ShagdarBinaryEvalConfig:
    device: str = "cpu"


class ShagdarBinaryEval:
    """Thin wrapper around Shagdar's released GCNMid binary head.

    Differs from `OoMixerForHB` / `GaberASIForHB` / `BaumannForHB` in that it
    does NOT output HB I..VI. Used as a binary comparator only.
    """

    def __init__(self, cfg: ShagdarBinaryEvalConfig | None = None):
        self.cfg = cfg or ShagdarBinaryEvalConfig()
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import sys
        shagdar_dir = (Path(__file__).resolve().parents[3]
                       / "src" / "baselines" / "shagdar_gnn")
        sys.path.insert(0, str(shagdar_dir))
        try:
            import graph_model  # noqa: WPS433
        finally:
            sys.path.pop(0)
        self._model = graph_model.GCNMid().to(self.cfg.device)

    def load_weights(self, weights_path: str | Path) -> None:
        self._ensure_loaded()
        state = torch.load(weights_path, map_location=self.cfg.device,
                           weights_only=False)
        self._model.load_state_dict(state)
        self._model.eval()

    def predict_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      weights: str | Path | None = None) -> torch.Tensor:
        """Return per-graph probability of `stroke` (class 1).

        Inputs are PyG-style tensors (node features, edge index, batch
        assignment). Build them with `src/baselines/shagdar_gnn/graphizer478.py`
        for our MediaPipe-478 landmarks.
        """
        if weights is not None:
            self.load_weights(weights)
        if self._model is None:
            raise RuntimeError("call load_weights() before predict_graph()")
        with torch.no_grad():
            logits = self._model(x.to(self.cfg.device),
                                 edge_index.to(self.cfg.device),
                                 batch.to(self.cfg.device))
            probs = torch.softmax(logits, dim=1)
        return probs[:, 1].cpu()
