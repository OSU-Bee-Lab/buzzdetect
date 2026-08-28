"""src/write/worker.py: how a chunk's results reach disk.

Results are appended to a _buzzpart.csv as chunks land, and only promoted to
_buzzdetect.csv -- sorted -- once the file is fully analyzed. That two-file
dance is what makes a run resumable, so it's worth exercising directly.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import tests._context as ctx  # noqa: F401

import src.config as cfg
from src.pipeline.assignments import AssignChunk, AssignFile
from src.write.worker import WorkerWriter

CLASSES = ['ambient_noise', 'ins_buzz', 'frog']
FRAMEHOP = 0.96


class WriterTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir_audio = os.path.join(self.tmp.name, 'audio')
        self.dir_out = os.path.join(self.tmp.name, 'out')
        os.makedirs(os.path.join(self.dir_audio, 'siteA'))
        self.a_file = AssignFile(
            path_audio=os.path.join(self.dir_audio, 'siteA', 'rec.wav'),
            dir_audio=self.dir_audio,
            dir_results=self.dir_out,
        )

    def writer(self, classes_out='all', threshold=None):
        return WorkerWriter(
            classes_out=classes_out, threshold=threshold, classes=CLASSES,
            framehop_s=FRAMEHOP, digits_time=2, dir_audio=self.dir_audio,
            dir_out=self.dir_out, digits_results=2, coordinator=mock.Mock(),
        )

    def chunk(self, start, n_frames=3):
        results = np.tile(np.array([0.1, 0.9, 0.3], dtype=np.float32), (n_frames, 1))
        return AssignChunk(file=self.a_file, chunk=(start, start + n_frames * FRAMEHOP),
                           results=results)


class TestPartialResults(WriterTestCase):
    def test_the_first_chunk_creates_the_partial_with_a_header(self):
        self.writer().write_results(self.chunk(0), fully_analyzed=False)
        self.assertTrue(os.path.exists(self.a_file.path_results_partial))
        self.assertFalse(os.path.exists(self.a_file.path_results_complete))
        df = pd.read_csv(self.a_file.path_results_partial)
        self.assertEqual(list(df.columns),
                         ['start', 'activation_ambient_noise', 'activation_ins_buzz', 'activation_frog'])

    def test_the_output_tree_mirrors_the_audio_tree(self):
        self.writer().write_results(self.chunk(0), fully_analyzed=False)
        self.assertTrue(os.path.exists(os.path.join(self.dir_out, 'siteA')))

    def test_later_chunks_append_without_repeating_the_header(self):
        w = self.writer()
        w.write_results(self.chunk(0), fully_analyzed=False)
        w.write_results(self.chunk(10), fully_analyzed=False)
        df = pd.read_csv(self.a_file.path_results_partial)
        self.assertEqual(len(df), 6)
        self.assertEqual(df['start'].dtype.kind, 'f')

    def test_timestamps_are_absolute_within_the_file(self):
        w = self.writer()
        w.write_results(self.chunk(100), fully_analyzed=False)
        df = pd.read_csv(self.a_file.path_results_partial)
        self.assertEqual(list(df['start']), [100.0, 100.96, 101.92])


class TestPromotion(WriterTestCase):
    def test_the_last_chunk_promotes_and_removes_the_partial(self):
        w = self.writer()
        w.write_results(self.chunk(0), fully_analyzed=True)
        self.assertTrue(os.path.exists(self.a_file.path_results_complete))
        self.assertFalse(os.path.exists(self.a_file.path_results_partial))

    def test_chunks_finishing_out_of_order_are_sorted_on_promotion(self):
        # Analyzers work in parallel, so a later chunk can land first.
        w = self.writer()
        w.write_results(self.chunk(100), fully_analyzed=False)
        w.write_results(self.chunk(0), fully_analyzed=False)
        w.write_results(self.chunk(50), fully_analyzed=True)
        df = pd.read_csv(self.a_file.path_results_complete)
        self.assertEqual(list(df['start']), sorted(df['start']))
        self.assertEqual(len(df), 9)

    def test_the_completed_name_is_the_one_the_skip_check_looks_for(self):
        self.writer().write_results(self.chunk(0), fully_analyzed=True)
        self.assertTrue(self.a_file.path_results_complete.endswith(cfg.SUFFIX_RESULT_COMPLETE))


class TestOutputModes(WriterTestCase):
    def test_activations_can_be_narrowed_to_selected_classes(self):
        w = self.writer(classes_out=['ins_buzz'])
        w.write_results(self.chunk(0), fully_analyzed=True)
        df = pd.read_csv(self.a_file.path_results_complete)
        self.assertEqual(list(df.columns), ['start', 'activation_ins_buzz'])

    def test_a_threshold_switches_the_schema_to_detections(self):
        w = self.writer(threshold=0.5)
        w.write_results(self.chunk(0), fully_analyzed=True)
        df = pd.read_csv(self.a_file.path_results_complete)
        self.assertEqual(list(df.columns), ['start', 'detections_ins_buzz'])
        self.assertEqual(set(df['detections_ins_buzz']), {1})


if __name__ == '__main__':
    unittest.main()
