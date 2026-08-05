from __future__ import annotations

import json
import unittest
from pathlib import Path


class PublishedResultContractTests(unittest.TestCase):
    def test_published_result_matches_the_frozen_110d_development_claim(self) -> None:
        path = Path(__file__).resolve().parents[1] / "results" / "current_development_model.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            "facial_paralysis_current_development_model_v1",
        )
        self.assertEqual(payload["model"]["feature_dimension"], 110)
        self.assertEqual(payload["model"]["classifier"]["c"], 0.01)
        self.assertAlmostEqual(
            payload["candidates"]["landmark_110d"]["auroc"],
            0.938375350140056,
        )
        self.assertFalse(payload["decision"]["outer_evaluation_authorized"])
        self.assertFalse(payload["decision"]["clinical_validation"])
        self.assertFalse(payload["decision"]["deployment_authorized"])


if __name__ == "__main__":
    unittest.main()
