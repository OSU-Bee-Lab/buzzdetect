"""src/write/formatting.py: turning a chunk's predictions into result rows.

The column names and the `start` timestamps here are the result schema the
manifest exists to protect, so they're asserted literally.
"""

import unittest

import numpy as np

import tests._context as ctx  # noqa: F401

from src.write.formatting import add_time, format_activations, format_detections

CLASSES = ['ambient_noise', 'ins_buzz', 'frog']
FRAMEHOP = 0.96


def results(rows):
    return np.array(rows, dtype=np.float32)


class TestAddTime(unittest.TestCase):
    def test_start_column_is_first_and_steps_by_framehop(self):
        import pandas as pd
        df = add_time(pd.DataFrame({'a': [0, 0, 0]}), 0, FRAMEHOP, 2)
        self.assertEqual(list(df.columns)[0], 'start')
        self.assertEqual(list(df['start']), [0.0, 0.96, 1.92])

    def test_offset_by_the_chunk_start(self):
        import pandas as pd
        df = add_time(pd.DataFrame({'a': [0, 0]}), 100.0, FRAMEHOP, 2)
        self.assertEqual(list(df['start']), [100.0, 100.96])

    def test_rounded_to_the_models_time_precision(self):
        import pandas as pd
        df = add_time(pd.DataFrame({'a': [0, 0]}), 0, 1 / 3, 2)
        self.assertEqual(list(df['start']), [0.0, 0.33])


class TestFormatDetections(unittest.TestCase):
    def test_thresholds_the_buzz_class_only(self):
        df = format_detections(results([[0.9, 0.2, 0.9], [0.1, 0.8, 0.1]]),
                               threshold=0.5, classes=CLASSES,
                               framehop_s=FRAMEHOP, digits_time=2, time_start=0)
        self.assertEqual(list(df.columns), ['start', 'detections_ins_buzz'])
        self.assertEqual(list(df['detections_ins_buzz']), [0, 1])

    def test_result_is_binary_integers(self):
        df = format_detections(results([[0, 0.6, 0]]), 0.5, CLASSES, FRAMEHOP, 2, 0)
        self.assertEqual(df['detections_ins_buzz'].dtype.kind, 'i')

    def test_exactly_at_the_threshold_is_not_a_detection(self):
        df = format_detections(results([[0, 0.5, 0]]), 0.5, CLASSES, FRAMEHOP, 2, 0)
        self.assertEqual(list(df['detections_ins_buzz']), [0])

    def test_a_model_without_a_buzz_class_is_an_error(self):
        with self.assertRaises(ValueError):
            format_detections(results([[0, 0]]), 0.5, ['frog', 'bird'], FRAMEHOP, 2, 0)


class TestFormatActivations(unittest.TestCase):
    def test_all_classes_by_default(self):
        df = format_activations(results([[0.11, 0.22, 0.33]]), CLASSES, FRAMEHOP, 2)
        self.assertEqual(list(df.columns),
                         ['start', 'activation_ambient_noise', 'activation_ins_buzz', 'activation_frog'])

    def test_values_are_rounded_to_the_requested_digits(self):
        df = format_activations(results([[0.126, 0.0, 0.0]]), CLASSES, FRAMEHOP, 2, digits_results=2)
        self.assertAlmostEqual(df['activation_ambient_noise'][0], 0.13, places=6)

    def test_subset_keeps_the_models_column_order_not_the_requests(self):
        df = format_activations(results([[0.1, 0.2, 0.3]]), CLASSES, FRAMEHOP, 2,
                                classes_keep=['frog', 'ambient_noise'])
        self.assertEqual(list(df.columns), ['start', 'activation_ambient_noise', 'activation_frog'])
        self.assertAlmostEqual(df['activation_frog'][0], 0.3, places=6)

    def test_unknown_class_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as e:
            format_activations(results([[0.1, 0.2, 0.3]]), CLASSES, FRAMEHOP, 2, classes_keep=['nope'])
        self.assertIn('nope', str(e.exception))

    def test_time_offset_applies(self):
        df = format_activations(results([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
                                CLASSES, FRAMEHOP, 2, time_start=10.0)
        self.assertEqual(list(df['start']), [10.0, 10.96])

    def test_input_array_is_not_mutated(self):
        r = results([[0.126, 0.0, 0.0]])
        format_activations(r, CLASSES, FRAMEHOP, 2)
        self.assertAlmostEqual(float(r[0][0]), 0.126, places=6)

    def test_class_list_is_not_mutated(self):
        classes = list(CLASSES)
        format_activations(results([[0.1, 0.2, 0.3]]), classes, FRAMEHOP, 2, classes_keep=['frog'])
        self.assertEqual(classes, CLASSES)

    def test_empty_result_still_has_the_schema(self):
        df = format_activations(np.zeros((0, 3), dtype=np.float32), CLASSES, FRAMEHOP, 2)
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns),
                         ['start', 'activation_ambient_noise', 'activation_ins_buzz', 'activation_frog'])


if __name__ == '__main__':
    unittest.main()
