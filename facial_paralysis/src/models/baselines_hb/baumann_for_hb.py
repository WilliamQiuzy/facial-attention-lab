"""Baumann 2022 → House-Brackmann adapter.

Baumann is the ONE baseline in our shortlist that natively outputs HB I..VI —
no head surgery, no retraining needed for the output format. The released
weights produce three competing HB grades per patient
(`grade_rowsum`, `grade_automata`, `grade_direct`), and the thesis itself
flags this multiple-output ambiguity as a methodological weakness.

This adapter:
  1. Wraps Baumann's `detect.py` pipeline so it can be invoked
     programmatically on a patient folder of 9 jpgs.
  2. Picks a single HB grade per patient using a configurable strategy.
  3. Returns I..VI as a Python int (1..6) for downstream comparison with
     other HB baselines.

**Retraining plan (locked 2026-05-26):** we fine-tune the released weights on
our HB-labeled patients once labels + per-pose segmentation arrive. Rationale:

  - The OTH n=86 weights likely overfit (thesis self-admission, see
    [[project-baumann-baseline]]). Don't trust them for direct inference.
  - Fine-tuning gives us a fair HB baseline on our cohort while still
    benefiting from ImageNet+OTH pretraining as initialization.
  - Output format stays HB I..VI throughout — no architectural surgery needed.

Use Baumann's `source/hbmedicalprocessing/train.py` (same `AUDIT-FIX` patches
required as for `detect.py`) with `hyp.yaml` (SGD 0.01 + cosine+exp LR +
Normalize(0.5)) as a starting point.

This adapter wraps the **inference** path; the fine-tuning path is run via
Baumann's `train.py` directly. After fine-tuning, point `weights_dir` at the
fine-tuned checkpoints and call `BaumannForHB` as usual.

**Default fusion choice:** `grade_direct`. Rationale: per Pass-2 audit
(`src/baselines/baumann_hb/AUDIT.md`), the thesis reports `Direct F1 = 1.000`
for Early Fusion (vs `Modulform F1 = 0.980`), and `grade_direct` is the only
strategy that bypasses the disagreement among the 4 module heads. It is also
the simplest interpretive story to a clinician ("the network said VI").

Input requirements (inherited from upstream):
  - One folder per patient containing 9 jpgs named `01.jpg` ... `09.jpg`
    (Baumann's own 9-pose order — NOT Knoedler; see AUDIT.md).
  - Pose names: rest / brow / smile_closed / smile_open / lip_pucker /
    eye_easy / eye_forced / nose_wrinkle / lip_depress.

Until per-pose segmentation lands (Task 3, clinician timestamps), you cannot
run Baumann on our LiveLinkFace `.mov` files directly. This adapter therefore
exposes the inference path; orchestration is left to a script.

Usage:
    from src.models.baselines_hb import BaumannForHB
    model = BaumannForHB(fusion="direct")
    grade = model.predict_patient("/path/to/patient_folder/")  # returns 1..6
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


FUSION = Literal["direct", "rowsum", "automata"]
HB_LABEL_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


@dataclass
class BaumannForHBConfig:
    fusion: FUSION = "direct"
    weights_dir: str = "src/baselines/baumann_hb/source/hbmedicalprocessing/models"
    device: str = "cpu"


class BaumannForHB:
    """Wrapper around Baumann's detect.py pipeline that returns one int HB grade.

    Not an `nn.Module` — Baumann's pipeline is a multi-stage script that runs
    face_alignment cropping + 5 ResNet-18 sub-models + fusion. Wrapping it as
    a module would require pulling all of that into PyTorch's autograd graph
    for no benefit. We treat it as a black-box predictor.
    """

    def __init__(self, cfg: BaumannForHBConfig | None = None):
        self.cfg = cfg or BaumannForHBConfig()
        # Defer the heavy import until predict() is called — this lets the
        # module be imported in environments that don't have face_alignment.
        self._run = None

    def _ensure_loaded(self) -> None:
        if self._run is not None:
            return
        import sys
        baumann_dir = (Path(__file__).resolve().parents[3]
                       / "src" / "baselines" / "baumann_hb"
                       / "source" / "hbmedicalprocessing")
        sys.path.insert(0, str(baumann_dir))
        try:
            from detect import run  # noqa: WPS433
        finally:
            sys.path.pop(0)
        self._run = run

    def predict_patient(self, patient_folder: str | Path) -> int:
        """Run Baumann's pipeline on one patient folder and return HB grade as int.

        Returns:
            1..6 corresponding to HB I..VI.

        Raises:
            ValueError if the folder doesn't have all 9 expected images or
            if Baumann's pipeline returns an unrecognized grade label.
        """
        self._ensure_loaded()
        patient_folder = Path(patient_folder)
        if not patient_folder.is_dir():
            raise ValueError(f"not a directory: {patient_folder}")
        for i in range(1, 10):
            if not (patient_folder / f"{i:02d}.jpg").exists():
                raise ValueError(f"missing {i:02d}.jpg in {patient_folder}")

        # Baumann's `run()` walks all category dirs under `source`. We pass a
        # one-level wrapper so a single patient folder is treated correctly.
        results = self._run(
            weights=self.cfg.weights_dir,
            source=str(patient_folder.parent),
            batch_size=1,
            device=self.cfg.device,
            half=False,
            function_selector="all",
            convert=True,
        )
        key = str(patient_folder)
        if key not in results:
            raise RuntimeError(f"patient key {key!r} not in Baumann output {list(results)}")
        label_field = f"grade_{self.cfg.fusion}"
        label = results[key].get(label_field)
        if label not in HB_LABEL_TO_INT:
            raise RuntimeError(f"unknown HB label {label!r} from Baumann")
        return HB_LABEL_TO_INT[label]
