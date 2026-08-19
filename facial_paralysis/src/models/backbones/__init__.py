import importlib

# Lazy (PEP 562): importing MARLIN (`marlin_video`) must not require `mediapipe`,
# which `oo_mlp_mixer` pulls in at module load. Import submodule attrs on demand.
_LAZY = {
    "OoMLPMixerEncoder": ".oo_mlp_mixer",
    "MarlinVideoEncoder": ".marlin_video",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(mod, __name__), name)


def __dir__():
    return sorted([*globals(), *_LAZY])


__all__ = ["OoMLPMixerEncoder", "MarlinVideoEncoder"]
