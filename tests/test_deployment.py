"""Check the files that must be present in a fresh deployment checkout."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import TrajectoryModel


class DeploymentTests(unittest.TestCase):
    def test_packaged_artifact_matches_release_metadata_and_r_predictions(self):
        artifact = ROOT / "artifacts" / "model.json.gz"
        self.assertTrue(artifact.is_file(), "The deployed checkout must include artifacts/model.json.gz")
        info = json.loads((ROOT / "model_info.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), info["artifact_sha256"])
        model = TrajectoryModel.load(artifact)
        self.assertEqual(len(model.trees), info["n_trees"])
        self.assertEqual(model.verify_export(), info["parity"]["case_count"])


if __name__ == "__main__":
    unittest.main()

