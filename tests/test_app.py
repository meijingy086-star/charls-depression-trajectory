import os
import sys
import unittest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)
from streamlit.testing.v1 import AppTest
from predictor import TrajectoryModel


class StreamlitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TrajectoryModel.load(os.path.join(APP_DIR, 'artifacts', 'model.json.gz'))

    def make_app(self):
        at = AppTest.from_file(os.path.join(APP_DIR, 'app.py'), default_timeout=30).run()
        self.assertFalse(at.exception, str(at.exception))
        return at

    def button(self, at, label):
        return next(b for b in at.button if b.label == label)

    def test_initial_render(self):
        at = self.make_app()
        self.assertEqual(len(at.selectbox), sum(s['type'] == 'category' for s in self.model.inputs))
        self.assertEqual(len(at.number_input), sum(s['type'] == 'number' for s in self.model.inputs))
        self.assertFalse(self.button(at, '预测轨迹类别').disabled)
        self.assertIsNone(at.session_state['result'])

    def test_missing_fields_do_not_generate_prediction(self):
        at = self.make_app()
        self.button(at, '预测轨迹类别').click().run()
        self.assertFalse(at.exception, str(at.exception))
        self.assertTrue(at.error)
        self.assertIsNone(at.session_state['result'])

    def test_example_predict_and_clear(self):
        at = self.make_app()
        self.button(at, '填入参考示例').click().run()
        self.assertFalse(at.exception, str(at.exception))
        self.button(at, '预测轨迹类别').click().run()
        self.assertFalse(at.exception, str(at.exception))
        result = at.session_state['result']
        self.assertIsNotNone(result)
        expected = self.model.predict(result['inputs'])
        self.assertEqual(result['class'], expected['class'])
        self.assertEqual(result['probabilities'], expected['probabilities'])
        self.assertAlmostEqual(sum(result['probabilities']), 1)
        self.button(at, '清空输入').click().run()
        self.assertFalse(at.exception, str(at.exception))
        self.assertIsNone(at.session_state['result'])
        self.assertTrue(all(s.value is None for s in at.selectbox))
        self.assertTrue(all(s.value is None for s in at.number_input))


if __name__ == '__main__':
    unittest.main()
