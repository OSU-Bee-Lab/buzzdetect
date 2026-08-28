"""src/validation.py: the argument checks both front ends run before a run.

Each validator returns ArgValid(valid, message); a valid result may still carry
a message, which is a warning rather than a refusal.
"""

import os
import tempfile
import unittest
import unittest.mock

import tests._context as ctx  # noqa: F401

import src.validation as v
from src.pipeline.loglevels import loglevels


class TestModelname(unittest.TestCase):
    def test_a_real_model_directory_passes(self):
        self.assertTrue(v.validate_modelname('model_general_v3').valid)

    def test_missing_model_is_named_in_the_message(self):
        result = v.validate_modelname('no_such_model')
        self.assertFalse(result.valid)
        self.assertIn('no_such_model', result.message)

    def test_a_directory_without_a_config_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'models', 'bare'))
            with unittest.mock.patch.object(v.cfg, 'DIR_MODELS', os.path.join(tmp, 'models')):
                self.assertFalse(v.validate_modelname('bare').valid)


class TestClassesOut(unittest.TestCase):
    def test_the_all_sentinel_passes(self):
        self.assertTrue(v.validate_classes_out('all').valid)

    def test_a_list_of_strings_passes(self):
        self.assertTrue(v.validate_classes_out(['ins_buzz']).valid)

    def test_a_bare_string_is_not_a_list(self):
        self.assertFalse(v.validate_classes_out('ins_buzz').valid)

    def test_non_string_members_fail(self):
        self.assertFalse(v.validate_classes_out(['ins_buzz', 3]).valid)

    def test_an_empty_list_is_not_a_usable_selection(self):
        # Nothing to write a column for; the frontend blocks Start on this too.
        self.assertFalse(v.validate_classes_out([]).valid)


class TestPrecision(unittest.TestCase):
    def test_none_means_activations_and_is_fine(self):
        self.assertTrue(v.validate_precision(None).valid)

    def test_ordinary_precision_passes_silently(self):
        result = v.validate_precision(0.95)
        self.assertTrue(result.valid)
        self.assertIsNone(result.message)

    def test_low_precision_passes_with_a_warning(self):
        result = v.validate_precision(0.5)
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.message)

    def test_out_of_range(self):
        for bad in (0, -1, 1, 1.5):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_precision(bad).valid)

    def test_not_a_number_is_refused(self):
        for bad in (float('nan'), float('inf')):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_precision(bad).valid)

    def test_non_numeric_is_refused_rather_than_raised(self):
        for bad in ('high', [], {}, object()):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_precision(bad).valid)


class TestFramehop(unittest.TestCase):
    def test_contiguous_frames_pass_silently(self):
        self.assertIsNone(v.validate_framehop(1).message)

    def test_overlap_passes(self):
        self.assertTrue(v.validate_framehop(0.5).valid)

    def test_gaps_pass_with_the_documented_warning(self):
        result = v.validate_framehop(2)
        self.assertTrue(result.valid)
        self.assertIn('missing data', result.message)

    def test_non_positive_is_refused(self):
        for bad in (0, -1):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_framehop(bad).valid)

    def test_non_numeric_is_refused_rather_than_raised(self):
        for bad in ('half', None, []):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_framehop(bad).valid)


class TestChunklength(unittest.TestCase):
    def test_int_and_float_pass(self):
        self.assertTrue(v.validate_chunklength(200).valid)
        self.assertTrue(v.validate_chunklength(0.5).valid)

    def test_non_positive_is_refused(self):
        self.assertFalse(v.validate_chunklength(0).valid)
        self.assertFalse(v.validate_chunklength(-5).valid)

    def test_non_numeric_is_refused_rather_than_raised(self):
        for bad in ('long', None, []):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_chunklength(bad).valid)


class TestInt(unittest.TestCase):
    def test_bounds_are_inclusive(self):
        self.assertTrue(v.validate_int(0, none_ok=False, value_min=0, value_max=1).valid)
        self.assertTrue(v.validate_int(1, none_ok=False, value_min=0, value_max=1).valid)
        self.assertFalse(v.validate_int(2, none_ok=False, value_min=0, value_max=1).valid)
        self.assertFalse(v.validate_int(-1, none_ok=False, value_min=0).valid)

    def test_none_is_gated_by_none_ok(self):
        self.assertTrue(v.validate_int(None, none_ok=True).valid)
        self.assertFalse(v.validate_int(None, none_ok=False).valid)

    def test_a_whole_number_float_is_an_integer(self):
        self.assertTrue(v.validate_int(3.0, none_ok=False).valid)

    def test_a_fractional_value_is_not_silently_truncated(self):
        self.assertFalse(v.validate_int(3.7, none_ok=False).valid)
        self.assertFalse(v.validate_int('3.7', none_ok=False).valid)

    def test_not_a_number_is_refused(self):
        for bad in (float('nan'), float('inf')):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_int(bad, none_ok=False).valid)

    def test_non_numeric_is_refused_rather_than_raised(self):
        for bad in ('two', [], {}):
            with self.subTest(bad=bad):
                self.assertFalse(v.validate_int(bad, none_ok=False).valid)


class TestWorkerCounts(unittest.TestCase):
    def test_cpu_analyzers_must_be_a_non_negative_count(self):
        self.assertTrue(v.validate_analyzers_cpu(0).valid)
        self.assertTrue(v.validate_analyzers_cpu(8).valid)
        self.assertFalse(v.validate_analyzers_cpu(-1).valid)
        self.assertFalse(v.validate_analyzers_cpu(None).valid)

    def test_gpu_analyzer_is_zero_or_one(self):
        self.assertTrue(v.validate_analyzer_gpu(1).valid)
        self.assertFalse(v.validate_analyzer_gpu(2).valid)

    def test_streamer_settings_may_be_unset(self):
        self.assertTrue(v.validate_n_streamers(None).valid)
        self.assertTrue(v.validate_stream_buffer_depth(None).valid)


class TestDirs(unittest.TestCase):
    def test_audio_dir_must_exist(self):
        self.assertTrue(v.validate_dir_audio(ctx.DIR_ENGINE).valid)
        result = v.validate_dir_audio(os.path.join(ctx.DIR_ENGINE, 'no_such_dir'))
        self.assertFalse(result.valid)
        self.assertIn('no_such_dir', result.message)

    def test_a_missing_output_dir_is_a_note_not_a_refusal(self):
        result = v.validate_dir_out(os.path.join(ctx.DIR_ENGINE, 'no_such_dir'))
        self.assertTrue(result.valid)
        self.assertIsNotNone(result.message)


class TestVerbosity(unittest.TestCase):
    def test_every_known_level_passes(self):
        for level in loglevels:
            with self.subTest(level=level):
                self.assertTrue(v.validate_verbosity(level).valid)

    def test_unknown_level_lists_the_valid_ones(self):
        result = v.validate_verbosity('CHATTY')
        self.assertFalse(result.valid)
        self.assertIn('DEBUG', result.message)


class TestValidateMap(unittest.TestCase):
    def test_every_entry_is_callable(self):
        for name, fn in v.validate_map.items():
            with self.subTest(name=name):
                self.assertTrue(callable(fn))

    def test_covers_every_analyze_parameter_the_cli_passes(self):
        # Keeps the map from silently falling behind analyze()'s signature.
        import inspect

        from src.analyze import analyze
        # q_gui and event_stopanalysis are wiring the legacy GUI hands in, not
        # settings a user types.
        params = set(inspect.signature(analyze).parameters) - {'q_gui', 'event_stopanalysis'}
        self.assertEqual(params - set(v.validate_map), set())


if __name__ == '__main__':
    import unittest.mock  # noqa: F401
    unittest.main()
