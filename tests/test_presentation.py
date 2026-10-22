import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from presentation import category_display_labels, class_display_names, input_display_name


class PresentationTests(unittest.TestCase):
    def test_old_artifact_does_not_invent_trajectory_names(self):
        self.assertEqual(class_display_names({}, ["Class1", "Class4"]),
                         {"Class1": "Class 1", "Class4": "Class 4"})

    def test_verified_labels_follow_class_keys(self):
        result = class_display_names({"class_labels": {"Class4": "Verified low trajectory"}},
                                     ["Class4", "Class1"])
        self.assertEqual(result["Class4"], "Verified low trajectory")
        self.assertEqual(result["Class1"], "Class 1")

    def test_new_three_level_input_is_supported(self):
        spec = {"name": "eyesight_distance", "options": [{"value": str(i)} for i in (1, 2, 3)]}
        self.assertEqual(category_display_labels(spec)["3"], "Poor")

    def test_unknown_code_uses_code_not_guessed_binary_meaning(self):
        spec = {"name": "new_factor", "options": [{"value": "3"}]}
        self.assertEqual(category_display_labels(spec), {"3": "Code 3"})

    def test_explicit_english_option_label_takes_precedence(self):
        spec = {"name": "gender", "options": [{"value": "1", "label_en": "Verified code one"}]}
        self.assertEqual(category_display_labels(spec)["1"], "Verified code one")

    def test_new_numeric_input_has_display_label(self):
        self.assertEqual(input_display_name({"name": "diasto"}), "Diastolic blood pressure")


if __name__ == "__main__":
    unittest.main()
