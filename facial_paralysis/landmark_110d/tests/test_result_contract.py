from __future__ import annotations

import json
import unittest
from pathlib import Path


class PublishedResultContractTests(unittest.TestCase):
    def test_published_result_matches_the_mirror_invariant_110d_claim(self) -> None:
        path = Path(__file__).resolve().parents[1] / "results" / "current_development_model.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema_version"],
            "facial_paralysis_current_development_model_v1",
        )
        self.assertEqual(payload["model"]["feature_dimension"], 110)
        self.assertEqual(
            payload["model"]["name"],
            "mirror_invariant_landmark_trajectory_110d_l2_logistic",
        )
        self.assertEqual(payload["model"]["classifier"]["c"], 0.01)
        self.assertAlmostEqual(
            payload["candidates"]["landmark_110d"]["auroc"],
            0.938375350140056,
        )
        self.assertAlmostEqual(
            payload["candidates"]["mirror_invariant_landmark_110d"]["auroc"],
            0.9439775910364145,
        )
        self.assertEqual(
            payload["decision"]["development_champion"],
            "mirror_invariant_landmark_110d",
        )
        self.assertTrue(
            payload["decision"]["mirror_robustness_successor_gate_passed"]
        )
        self.assertFalse(payload["decision"]["outer_evaluation_authorized"])
        self.assertFalse(payload["decision"]["clinical_validation"])
        self.assertFalse(payload["decision"]["deployment_authorized"])


if __name__ == "__main__":
    unittest.main()
