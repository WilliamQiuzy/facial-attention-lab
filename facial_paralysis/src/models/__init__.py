"""Models for facial-paralysis severity grading.

See `docs/model_design.md` for the architecture. New work should use the
multi-task severity model; `HBHead` is the legacy single-task head kept for the
existing smoke test.
"""
from src.models.ordinal import (
    OrderedThresholds,
    OrdinalThresholdHead,
    class_probs,
    cum_probs,
    expected_grade,
    ordinal_loss,
    predict_grade,
)
from src.models.multitask import (
    DEFAULT_TASKS,
    MultiTaskSeverityModel,
    SeverityTrunk,
    TaskSpec,
    TrunkConfig,
    multitask_loss,
)
from src.models.temporal import TemporalLandmarkEncoder
from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
from src.models.hb_head import HBHead, HBHeadConfig

__all__ = [
    # full pipeline
    "FacialPalsyModel",
    "FacialPalsyConfig",
    "TemporalLandmarkEncoder",
    # ordinal primitives
    "OrderedThresholds",
    "OrdinalThresholdHead",
    "ordinal_loss",
    "class_probs",
    "cum_probs",
    "predict_grade",
    "expected_grade",
    # multi-task model
    "MultiTaskSeverityModel",
    "SeverityTrunk",
    "TaskSpec",
    "TrunkConfig",
    "DEFAULT_TASKS",
    "multitask_loss",
    # legacy
    "HBHead",
    "HBHeadConfig",
]
