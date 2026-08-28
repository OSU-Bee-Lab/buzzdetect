"""src/stream/audio.py: picking a reader for a file.

Our own drivers take precedence over soundfile's claimed format support, which
is the point of the module -- soundfile claims mp3 and cannot read one to the
end (see drivers/mp3.py).
"""

import os
import tempfile
import unittest

import tests._context as ctx  # noqa: F401

import soundfile as sf

import src.config as cfg
from src.pipeline.assignments import AssignFile
from src.stream.audio import UnsupportedFormat, build_track, driver_map, get_duration
from src.stream.driver import AudioDriver


class TestDriverMap(unittest.TestCase):
    def test_every_driver_in_the_drivers_directory_is_registered(self):
        for name in os.listdir(cfg.DIR_DRIVERS):
            if not name.endswith('.py') or name.startswith('_'):
                continue
            with self.subTest(driver=name):
                self.assertIn(os.path.splitext(name)[0].lower(), driver_map)

    def test_our_driver_wins_over_soundfiles_claimed_support(self):
        # soundfile does claim mp3; it just can't read one to the end.
        self.assertIn('MP3', sf.available_formats())
        self.assertIsNot(driver_map['mp3'], sf.SoundFile)

    def test_soundfile_is_the_fallback_for_formats_we_have_no_driver_for(self):
        self.assertIs(driver_map['wav'], sf.SoundFile)

    def test_keys_are_lowercase_extensions(self):
        for key in driver_map:
            with self.subTest(key=key):
                self.assertEqual(key, key.lower())

    def test_every_driver_offers_what_the_streamer_calls(self):
        # The drivers are loaded off disk by path and don't subclass
        # AudioDriver, so the interface it declares is honoured by duck typing;
        # this is what checks that it still is.
        interface = [name for name in vars(AudioDriver) if not name.startswith('_')]
        self.assertEqual(sorted(interface), ['close', 'read', 'seek', 'tell'])
        for ext, driver in driver_map.items():
            if driver is sf.SoundFile:
                continue
            with self.subTest(ext=ext):
                for name in interface:
                    self.assertTrue(callable(getattr(driver, name, None)),
                                    f'{ext} driver has no {name}()')


class TestBuildTrack(unittest.TestCase):
    def test_an_mp3_gets_our_driver(self):
        track = build_track(ctx.fixture('tiny.mp3'))
        try:
            self.assertIs(type(track), driver_map['mp3'])
            self.assertIsNot(type(track), sf.SoundFile)
            for attr in ('samplerate', 'channels', 'frames'):
                self.assertIsNotNone(getattr(track, attr))
        finally:
            track.close()

    def test_extension_case_does_not_matter(self):
        with tempfile.TemporaryDirectory() as tmp:
            upper = os.path.join(tmp, 'REC.MP3')
            with open(ctx.fixture('tiny.mp3'), 'rb') as src, open(upper, 'wb') as dst:
                dst.write(src.read())
            track = build_track(upper)
            track.close()

    def test_an_unsupported_format_names_the_extension(self):
        with self.assertRaises(UnsupportedFormat) as e:
            build_track('/audio/notes.txt')
        self.assertIn('txt', str(e.exception))


class TestGetDuration(unittest.TestCase):
    def a_file(self, path):
        return AssignFile(path_audio=path, dir_audio=os.path.dirname(path), dir_results='/out')

    def test_opens_the_track_if_it_is_not_open_yet(self):
        a = self.a_file(ctx.fixture('truncating.mp3'))
        duration = get_duration(a)
        try:
            self.assertIsNotNone(a.track)
            self.assertGreater(duration, 0)
            self.assertAlmostEqual(duration, a.track.frames / a.track.samplerate, places=6)
        finally:
            a.track.close()

    def test_a_stereo_file_reports_its_length_not_its_sample_count(self):
        a = self.a_file(ctx.fixture('stereo.mp3'))
        duration = get_duration(a)
        try:
            self.assertGreater(a.track.channels, 1)
            self.assertGreater(duration, 0)
            self.assertLess(duration, 600)
        finally:
            a.track.close()

    def test_a_wav_goes_through_soundfile(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tone.wav')
            sf.write(path, np.zeros(16000 * 3, dtype='float32'), 16000)
            a = self.a_file(path)
            self.assertAlmostEqual(get_duration(a), 3.0, places=6)
            a.track.close()


if __name__ == '__main__':
    unittest.main()
