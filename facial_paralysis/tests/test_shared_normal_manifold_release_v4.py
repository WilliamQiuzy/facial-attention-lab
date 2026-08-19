from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from _testlib import run_all

ARTIFACT=ROOT/"docs/results/artifacts/shared_normal_manifold_router_v4"


def test_all_machine_reports_are_parseable_and_protected_closed(c):
    names={"screen-seed0.json","v5-screen-seed0.json","v5-confirm-seed1.json","v5-confirm-seed2.json","v6-screen-seed0.json","v6-confirm-seed1.json","v6-confirm-seed2.json","v7-screen-seed0.json","v8-screen-seed0.json","v9-screen-seed0.json"}
    c.eq({path.name for path in ARTIFACT.glob("*.json")},names)
    for path in ARTIFACT.glob("*.json"):
        payload=path.read_bytes(); report=json.loads(payload)
        c.eq(report["audit"]["mayo_reads"],0)
        c.eq(report["audit"]["palsynet_protected_reads"],0)
        text=payload.decode().lower()
        c.true("/users/" not in text and "/home/" not in text)
        c.true("group_id" not in text and "participant_id" not in text)


def test_failed_search_did_not_change_canonical_ucr4(c):
    expected={
        "docs/model_registry.json":"d41843e0f8e32fd9e2f7fa2451c4b9d996e3433ffcb417c15275a63bebef333b",
        "docs/CURRENT_MODEL.md":"50842973631847f6aed8fe7100f05ae9d2d41ceabe7adb0d2fd1f7f8e1299e56",
        "src/models/current.py":"506f7b97c948c4f9e919981a6a771d9c03ee0194752f95f5912e65552e1fc8c4",
    }
    for relative,digest in expected.items():
        c.eq(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),digest)
    registry=json.loads((ROOT/"docs/model_registry.json").read_text())
    c.eq(registry["current"]["name"],"universal_clinical_router_v4")


if __name__=="__main__": run_all("test_shared_normal_manifold_release_v4",dict(globals()))
