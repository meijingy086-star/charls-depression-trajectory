import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from predictor import ModelError, TrajectoryModel


def fixture():
    # Synthetic test trees only; these are never used as the study model.
    return {
        "format": "charls-rf-v1", "classes": ["Class1","Class2","Class3","Class4"],
        "n_trees": 2,
        "inputs": [
            {"name":"sleep","label":"睡眠时间","type":"number","observed_min":2,"observed_max":10},
            {"name":"gender","label":"性别","type":"category","options":[{"value":"1","label":"女"},{"value":"2","label":"男"}]}
        ],
        "features": [{"name":"sleep","type":"number","original":"sleep"},
                     {"name":"gender.2","type":"category","original":"gender","values":{"1":0,"2":1}}],
        "trees": [
            [[2,3,1,6,1,0],[0,0,0,0,-1,1],[0,0,0,0,-1,4]],
            [[2,3,2,.5,1,0],[0,0,0,0,-1,2],[0,0,0,0,-1,3]]
        ],
        "verification_cases": [{"inputs":{"sleep":6,"gender":"1"},"encoded":[6,0],"probabilities":[.5,.5,0,0]}]
    }


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.model = TrajectoryModel(fixture())

    def test_threshold_is_less_than_or_equal(self):
        left = self.model.predict({"sleep":6,"gender":"1"})
        right = self.model.predict({"sleep":6.00000001,"gender":"1"})
        self.assertEqual(left["probabilities"],[.5,.5,0,0])
        self.assertEqual(right["probabilities"],[0,.5,0,.5])

    def test_encoding_matches_export(self):
        self.assertEqual(self.model.encode({"sleep":7,"gender":"2"})[0],[7,1])
        self.assertEqual(self.model.verify_export(),1)

    def test_ties_use_first_probability_column(self):
        result = self.model.predict({"sleep":8,"gender":"2"})
        self.assertEqual(result["class"],"Class3")
        self.assertTrue(result["tie"])

    def test_preserves_r_class_column_order(self):
        data = fixture()
        data["classes"] = ["Class4","Class3","Class2","Class1"]
        model = TrajectoryModel(data)
        self.assertEqual(model.predict({"sleep":6,"gender":"1"})["class"],"Class4")

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(self.model.predict({"sleep":7,"gender":"2"})["probabilities"]),1)

    def test_missing_inputs_fail(self):
        for raw in ({"gender":"1"},{"sleep":5,"gender":None},{"sleep":None,"gender":"2"}):
            with self.assertRaises(ModelError): self.model.predict(raw)

    def test_unknown_categories_fail(self):
        for category in ("0","3","男",True):
            with self.assertRaises(ModelError): self.model.predict({"sleep":7,"gender":category})

    def test_nonfinite_values_fail(self):
        for value in (float("nan"),float("inf"),"abc",True):
            with self.assertRaises(ModelError): self.model.predict({"sleep":value,"gender":"1"})

    def test_outside_training_range_warns(self):
        self.assertEqual(len(self.model.predict({"sleep":12,"gender":"1"})["warnings"]),1)

    def test_cycles_are_rejected(self):
        data = fixture()
        data["trees"][0][0][0] = 1
        with self.assertRaises(ModelError): TrajectoryModel(data)

    def test_invalid_leaf_class_is_rejected(self):
        data = fixture()
        data["trees"][0][1][5] = 5
        with self.assertRaises(ModelError): TrajectoryModel(data)

    def test_single_leaf_tree(self):
        data = fixture()
        data["trees"] = [[[0,0,0,0,-1,4]]]
        data["n_trees"] = 1
        model = TrajectoryModel(data)
        self.assertEqual(model.predict({"sleep":5,"gender":"1"})["probabilities"],[0,0,0,1])

    def test_gzip_roundtrip_and_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json.gz"
            with gzip.open(str(path),"wt",encoding="utf-8") as handle: json.dump(fixture(),handle)
            loaded = TrajectoryModel.load(path)
            self.assertEqual(len(loaded.digest),64)
            self.assertEqual(loaded.verify_export(),1)

    def test_corrupt_expected_predictions_fail_load(self):
        data = fixture()
        data["verification_cases"][0]["probabilities"] = [1,0,0,0]
        with self.assertRaises(ModelError): TrajectoryModel(data).verify_export()

    def test_corrupt_expected_encoding_fails(self):
        data = fixture()
        data["verification_cases"][0]["encoded"] = [6,1]
        with self.assertRaises(ModelError): TrajectoryModel(data).verify_export()

    def test_incomplete_mapping_fails(self):
        data = fixture()
        del data["features"][1]["values"]["2"]
        with self.assertRaises(ModelError): TrajectoryModel(data)


if __name__ == "__main__":
    unittest.main()
