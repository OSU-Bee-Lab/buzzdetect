"""src/utils.py: extension parsing, tree walking, identity building."""

import os
import tempfile
import unittest

import tests._context as ctx  # noqa: F401  (chdir + sys.path)

from src.utils import Timer, build_ident, get_ext, search_dir


class TestGetExt(unittest.TestCase):
    def test_lowercases_and_strips_dot(self):
        self.assertEqual(get_ext('a/b/c.WAV'), 'wav')
        self.assertEqual(get_ext('recording.mp3'), 'mp3')

    def test_no_extension(self):
        self.assertEqual(get_ext('a/b/README'), '')

    def test_dotfile_is_not_an_extension(self):
        # os.path.splitext treats a leading-dot name as all-stem, which is what
        # we want: '.gitignore' is not a '.gitignore-format' audio file.
        self.assertEqual(get_ext('.gitignore'), '')

    def test_only_the_last_extension(self):
        self.assertEqual(get_ext('a.tar.gz'), 'gz')


class TestSearchDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, 'siteA', 'nested'))
        for rel in ('top.wav', 'top.txt',
                    'siteA/a.mp3', 'siteA/b.WAV',
                    'siteA/nested/c.wav'):
            path = os.path.join(self.root, rel)
            with open(path, 'w') as f:
                f.write('x')

    def tearDown(self):
        self.tmp.cleanup()

    def found(self, extensions=None):
        return sorted(os.path.relpath(p, self.root).replace(os.sep, '/')
                      for p in search_dir(self.root, extensions))

    def test_yields_lazily(self):
        # A generator, not a list: the point of the function is that a caller
        # can act on the first match before the walk has finished.
        gen = search_dir(self.root)
        self.assertTrue(hasattr(gen, '__next__'))

    def test_no_filter_returns_every_file(self):
        self.assertEqual(self.found(), [
            'siteA/a.mp3', 'siteA/b.WAV', 'siteA/nested/c.wav',
            'top.txt', 'top.wav',
        ])

    def test_filters_by_extension_case_insensitively(self):
        self.assertEqual(self.found(['wav']), ['siteA/b.WAV', 'siteA/nested/c.wav', 'top.wav'])

    def test_multiple_extensions(self):
        self.assertEqual(self.found(['wav', 'mp3']),
                         ['siteA/a.mp3', 'siteA/b.WAV', 'siteA/nested/c.wav', 'top.wav'])

    def test_rejects_bad_extension_argument(self):
        for bad in ('wav', [1], ()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    list(search_dir(self.root, bad))

    def test_empty_extension_list_matches_nothing(self):
        self.assertEqual(self.found([]), [])

    def test_missing_directory_yields_nothing(self):
        self.assertEqual(list(search_dir(os.path.join(self.root, 'nope'))), [])


class TestBuildIdent(unittest.TestCase):
    def test_strips_root_extension_and_leading_slash(self):
        self.assertEqual(build_ident('/data/audio/siteA/rec.wav', '/data/audio'), 'siteA/rec.wav'[:-4])

    def test_file_at_root(self):
        self.assertEqual(build_ident('/data/audio/rec.wav', '/data/audio'), 'rec')

    def test_tag_is_removed(self):
        self.assertEqual(build_ident('/data/audio/rec_raw.wav', '/data/audio', tag='_raw'), 'rec')

    def test_root_with_regex_metacharacters(self):
        # The root directory is a plain path, not a pattern: a '+' or '(' in a
        # folder name must not be read as a quantifier or a group.
        self.assertEqual(build_ident('/data/audio+raw/rec.wav', '/data/audio+raw'), 'rec')
        self.assertEqual(build_ident('/data/a.b/rec.wav', '/data/a.b'), 'rec')

    def test_root_substring_is_not_removed_from_elsewhere(self):
        # 'audio' also appears inside the filename; only the leading root goes.
        self.assertEqual(build_ident('/data/audio/audio_2.wav', '/data/audio'), 'audio_2')


class TestTimer(unittest.TestCase):
    def test_total_is_rounded_seconds_between_start_and_stop(self):
        t = Timer()
        t.stop()
        self.assertIsInstance(t.get_total(), float)
        self.assertGreaterEqual(t.get_total(), 0)

    def test_current_advances_from_start(self):
        t = Timer()
        self.assertGreaterEqual(t.get_current().total_seconds(), 0)


if __name__ == '__main__':
    unittest.main()
