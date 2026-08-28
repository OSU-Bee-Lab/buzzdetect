"""src/pipeline/assignments.py: the shapes passed between the workers.

AssignFile derives every path a file's results are written to, so its
arithmetic is what decides where output lands and what the GUI displays.
"""

import unittest

import tests._context as ctx  # noqa: F401

import src.config as cfg
from src.pipeline.assignments import AssignChunk, AssignFile, AssignLog
from src.pipeline.loglevels import loglevels


def assign(path='/audio/siteA/rec.wav', dir_audio='/audio', dir_results='/out'):
    return AssignFile(path_audio=path, dir_audio=dir_audio, dir_results=dir_results)


class TestAssignFile(unittest.TestCase):
    def test_ident_is_the_path_below_the_audio_directory(self):
        self.assertEqual(assign().ident, 'siteA/rec')

    def test_results_mirror_the_audio_tree_under_the_output_directory(self):
        a = assign()
        self.assertEqual(a.path_results_complete, '/out/siteA/rec' + cfg.SUFFIX_RESULT_COMPLETE)
        self.assertEqual(a.path_results_partial, '/out/siteA/rec' + cfg.SUFFIX_RESULT_PARTIAL)

    def test_partial_and_complete_are_different_files(self):
        a = assign()
        self.assertNotEqual(a.path_results_partial, a.path_results_complete)

    def test_extension_is_kept_for_the_display_path_only(self):
        a = assign()
        self.assertEqual(a.extension_audio, '.wav')
        self.assertEqual(a.shortpath_audio, 'siteA/rec.wav')
        self.assertEqual(a.shortpath_results_complete, 'siteA/rec' + cfg.SUFFIX_RESULT_COMPLETE)

    def test_a_file_at_the_root_of_the_audio_directory(self):
        a = assign(path='/audio/rec.mp3')
        self.assertEqual(a.ident, 'rec')
        self.assertEqual(a.shortpath_audio, 'rec.mp3')

    def test_an_audio_directory_with_a_regex_metacharacter_in_its_name(self):
        a = assign(path='/audio+raw/siteA/rec.wav', dir_audio='/audio+raw')
        self.assertEqual(a.ident, 'siteA/rec')

    def test_two_files_differing_only_by_extension_collide(self):
        # Documented behaviour, not an accident: the ident drops the extension,
        # so a.wav and a.mp3 want the same result file. analyze() checks for
        # this and skips the second with reason 'name_conflict'.
        self.assertEqual(assign(path='/audio/a.wav').ident, assign(path='/audio/a.mp3').ident)

    def test_starts_with_nothing_read(self):
        a = assign()
        self.assertIsNone(a.track)
        self.assertIsNone(a.duration_audio)
        self.assertIsNone(a.chunklist)


class TestAssignChunk(unittest.TestCase):
    def test_carries_its_file_and_defaults_to_not_being_the_last(self):
        c = AssignChunk(file=assign(), chunk=(0.0, 200.0))
        self.assertFalse(c.last_chunk)
        self.assertIsNone(c.samples)
        self.assertIsNone(c.results)
        self.assertEqual(c.file.ident, 'siteA/rec')


class TestAssignLog(unittest.TestCase):
    def test_level_string_is_resolved_to_its_number(self):
        self.assertEqual(AssignLog('hi', 'DEBUG').level_int, loglevels['DEBUG'])

    def test_an_unknown_level_fails_at_the_point_it_is_written(self):
        with self.assertRaises(KeyError):
            AssignLog('hi', 'CHATTY')

    def test_levels_are_ordered_least_to_most_severe(self):
        values = list(loglevels.values())
        self.assertEqual(values, sorted(values))

    def test_terminate_is_off_by_default(self):
        self.assertFalse(AssignLog('hi', 'INFO').terminate)


if __name__ == '__main__':
    unittest.main()
