"""The .m4a driver: a thin alias of the mp4 driver, checked on a generated file."""

import os
import tempfile
import unittest

import tests._context as ctx  # noqa: F401

import av
import numpy as np

from src.stream.audio import build_track, driver_map

RATE = 16000
SECONDS = 6


def write_m4a(path, codec='aac'):
    t = np.arange(RATE * SECONDS) / RATE
    pcm = (0.4 * np.sin(2 * np.pi * 440 * t) * np.sin(2 * np.pi * 0.7 * t)).astype(np.float32)
    with av.open(path, 'w', format='ipod') as out:
        stream = out.add_stream(codec, rate=RATE)
        stream.layout = 'mono'
        frame = av.AudioFrame.from_ndarray(pcm[None, :], format='flt', layout='mono')
        frame.sample_rate = RATE
        frame.pts = 0
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


class TestM4a(unittest.TestCase):
    def test_registered_to_the_mp4_driver(self):
        self.assertIs(driver_map['m4a'], driver_map['mp4'])

    def test_read_and_seek_agree_with_a_linear_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'rec.m4a')
            write_m4a(path)

            track = build_track(path)
            try:
                self.assertEqual(track.samplerate, RATE)
                self.assertEqual(track.channels, 1)
                whole = track.read(RATE * SECONDS)
                self.assertGreater(len(whole), RATE * (SECONDS - 1))

                for start in (RATE * 4, 777, RATE * 2 + 5):  # backward seeks
                    track.seek(start)
                    self.assertEqual(track.tell(), start)
                    got = track.read(RATE)
                    np.testing.assert_allclose(got, whole[start:start + RATE], atol=1e-6)
            finally:
                track.close()


if __name__ == '__main__':
    unittest.main()
