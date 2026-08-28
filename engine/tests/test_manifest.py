"""src/pipeline/manifest.py: the output folder's schema lock.

Both sides of the app enforce this independently -- Python here, the Svelte
frontend in checkManifest() -- so the rules it encodes are worth pinning down.
"""

import json
import os
import tempfile
import unittest

import tests._context as ctx  # noqa: F401

from src.pipeline import manifest as mf


def activations(classes=('ins_buzz', 'frog'), modelname='m', framehop_prop=1):
    return mf.build_manifest(modelname, framehop_prop, None, list(classes))


def detections(precision=0.95, modelname='m', framehop_prop=1):
    return mf.build_manifest(modelname, framehop_prop, precision, 'all')


class TestBuildManifest(unittest.TestCase):
    def test_no_precision_means_activations(self):
        m = activations()
        self.assertEqual(m['output_mode'], 'activations')
        self.assertIsNone(m['precision'])

    def test_precision_means_detections(self):
        m = detections()
        self.assertEqual(m['output_mode'], 'detections')
        # The class list doesn't shape a detections result, so it isn't locked.
        self.assertIsNone(m['classes_out'])

    def test_classes_are_sorted_so_selection_order_does_not_matter(self):
        self.assertEqual(activations(['b', 'a', 'c'])['classes_out'], ['a', 'b', 'c'])

    def test_records_every_locked_key(self):
        self.assertEqual(set(activations()), set(mf.KEYS_LOCKED) | {'output_mode'})


class TestReadWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir_out = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_manifest_reads_as_none(self):
        self.assertIsNone(mf.read_manifest(self.dir_out))

    def test_round_trip(self):
        m = activations()
        mf.write_manifest(self.dir_out, m)
        self.assertEqual(mf.read_manifest(self.dir_out), m)

    def test_write_creates_the_folder(self):
        nested = os.path.join(self.dir_out, 'a', 'b')
        mf.write_manifest(nested, activations())
        self.assertTrue(os.path.exists(os.path.join(nested, mf.FNAME_MANIFEST)))

    def test_written_file_is_readable_json_at_the_documented_name(self):
        mf.write_manifest(self.dir_out, activations())
        with open(os.path.join(self.dir_out, 'buzzdetect_manifest.json')) as f:
            self.assertEqual(json.load(f)['modelname'], 'm')


class TestDiff(unittest.TestCase):
    def test_identical_manifests_do_not_conflict(self):
        self.assertEqual(mf.diff_manifests(activations(), activations()), [])

    def test_class_order_is_not_a_conflict(self):
        self.assertEqual(mf.diff_manifests(activations(['a', 'b']), activations(['b', 'a'])), [])

    def test_class_set_change_is_reported_both_ways(self):
        conflicts = mf.diff_manifests(activations(['a', 'b']), activations(['b', 'c']))
        self.assertEqual(len(conflicts), 1)
        self.assertIn('added c', conflicts[0])
        self.assertIn('removed a', conflicts[0])

    def test_model_change_conflicts(self):
        conflicts = mf.diff_manifests(activations(modelname='old'), activations(modelname='new'))
        self.assertTrue(any('modelname' in c for c in conflicts))

    def test_mode_change_conflicts(self):
        conflicts = mf.diff_manifests(activations(), detections())
        self.assertTrue(any('output_mode' in c for c in conflicts))

    def test_framehop_change_conflicts(self):
        conflicts = mf.diff_manifests(activations(framehop_prop=1), activations(framehop_prop=0.5))
        self.assertTrue(any('framehop_prop' in c for c in conflicts))

    def test_every_locked_key_is_actually_checked(self):
        for key in mf.KEYS_LOCKED:
            with self.subTest(key=key):
                current = dict(activations())
                current[key] = ['sentinel'] if key == 'classes_out' else 'sentinel'
                self.assertTrue(mf.diff_manifests(activations(), current))

    def test_a_key_outside_the_locked_set_is_ignored(self):
        current = dict(activations())
        current['something_new'] = 'value'
        self.assertEqual(mf.diff_manifests(activations(), current), [])


class TestCheckOrWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir_out = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_folder_is_claimed(self):
        ok, msg = mf.check_or_write_manifest(self.dir_out, activations())
        self.assertTrue(ok)
        self.assertIsNone(msg)
        self.assertEqual(mf.read_manifest(self.dir_out), activations())

    def test_matching_settings_pass(self):
        mf.check_or_write_manifest(self.dir_out, activations())
        ok, msg = mf.check_or_write_manifest(self.dir_out, activations(['frog', 'ins_buzz']))
        self.assertTrue(ok)
        self.assertIsNone(msg)

    def test_conflicting_settings_are_refused_and_nothing_is_overwritten(self):
        mf.check_or_write_manifest(self.dir_out, activations())
        ok, msg = mf.check_or_write_manifest(self.dir_out, detections())
        self.assertFalse(ok)
        self.assertIn('output_mode', msg)
        self.assertEqual(mf.read_manifest(self.dir_out), activations())


if __name__ == '__main__':
    unittest.main()
