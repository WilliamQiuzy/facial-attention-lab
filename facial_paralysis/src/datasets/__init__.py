"""Dataset exports without forcing the optional PyTorch training runtime."""
from __future__ import annotations

from importlib import import_module

__all__ = ["PatientVideoDataset", "PatientRecord", "STANDARD_ACTIONS"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".patient_videos", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
