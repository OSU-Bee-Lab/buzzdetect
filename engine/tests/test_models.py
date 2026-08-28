"""src/inference/models.py: the framing arithmetic and the fixed-length session.

n_frames decides how much of the graph's output is real audio rather than
padding, so being wrong by one frame here either drops a second of audio off
every chunk or writes a frame of silence into the results. The graph itself is
the oracle: the last class of tests builds a session at a given input length
and checks the row count the graph actually returns.
"""

import unittest

import numpy as np

import tests._context as ctx  # noqa: F401

from src.inference.models import OnnxModel, load_model

MODEL = 'model_general_v3'


def model(framehop_prop=1):
    return load_model(MODEL, framehop_prop=framehop_prop, initialize=False)


class TestConstruction(unittest.TestCase):
    def test_loads_the_class_defined_in_the_models_own_model_py(self):
        # Not the OnnxModel it imports to subclass, which sorts first in dir().
        m = model()
        self.assertIsInstance(m, OnnxModel)
        self.assertIsNot(type(m), OnnxModel)
        self.assertEqual(m.modelname, MODEL)

    def test_reads_its_class_list(self):
        self.assertIn('ins_buzz', model().config['classes'])

    def test_unknown_model_is_refused_by_name(self):
        with self.assertRaises(ValueError) as e:
            load_model('no_such_model', framehop_prop=1, initialize=False)
        self.assertIn('no_such_model', str(e.exception))

    def test_overlapping_frames_are_refused_not_ignored(self):
        # The patch hop is welded into the exported graph, so an overlapping
        # framehop can't be honoured and must not be silently accepted.
        with self.assertRaises(ValueError):
            model(framehop_prop=0.5)

    def test_framehop_seconds_follow_the_frame_length(self):
        m = model()
        self.assertEqual(m.framehop_s, m.framelength_s)


class TestNFrames(unittest.TestCase):
    def setUp(self):
        self.m = model()

    def test_no_audio_is_no_frames(self):
        self.assertEqual(self.m.n_frames(0), 0)
        self.assertEqual(self.m.n_frames(-1), 0)

    def test_anything_shorter_than_one_frame_still_makes_one(self):
        # The front end pads a short input up to a whole patch.
        for n in (1, 1000, self.m.samples_min - 1, self.m.samples_min):
            with self.subTest(n=n):
                self.assertEqual(self.m.n_frames(n), 1)

    def test_one_sample_past_a_full_frame_starts_the_next(self):
        self.assertEqual(self.m.n_frames(self.m.samples_min + 1), 2)

    def test_frames_never_go_backwards_as_audio_grows(self):
        counts = [self.m.n_frames(n) for n in range(0, 200000, 997)]
        self.assertEqual(counts, sorted(counts))

    def test_the_float32_reciprocal_is_load_bearing(self):
        # tf2onnx emits the hop division as a float32 multiply by the
        # reciprocal, and 1/15360 isn't exact in float32: at these lengths the
        # quotient lands just above the integer and the graph returns one more
        # frame than exact arithmetic would. Doing it in float64 here would
        # silently drop that frame.
        for n in (61680, 3210480):
            with self.subTest(n=n):
                after = n - self.m.samples_min
                exact = 1 + int(np.ceil(after / self.m.samples_hop))
                self.assertEqual(self.m.n_frames(n), exact + 1)


class TestSessionLength(unittest.TestCase):
    def setUp(self):
        self.m = model()

    def test_holds_a_whole_chunk_with_room_to_spare(self):
        # Resampling doesn't always land on the exact expected sample count, so
        # the session is deliberately sized past the chunk: a chunk that comes
        # back a few samples long still has to fit the session it was sized for.
        for chunklength_s in (1, 10, 200, 200.5):
            with self.subTest(chunklength_s=chunklength_s):
                n_chunk = int(round(chunklength_s * self.m.samplerate))
                length = self.m.session_length(chunklength_s)
                self.assertGreater(length, n_chunk)
                self.assertGreater(self.m.n_frames(length), self.m.n_frames(n_chunk))

    def test_is_a_whole_number_of_hops_past_the_first_frame(self):
        length = self.m.session_length(200)
        self.assertEqual((length - self.m.samples_min) % self.m.samples_hop, 0)


class TestPredict(unittest.TestCase):
    """Padding and trimming, against a real session."""

    @classmethod
    def setUpClass(cls):
        cls.chunklength_s = 5
        cls.m = model()
        cls.m.samples_session = cls.m.session_length(cls.chunklength_s)
        cls.m.initialize()

    def test_a_full_chunk_returns_a_row_per_frame_and_a_column_per_class(self):
        n = self.chunklength_s * self.m.samplerate
        out = self.m.predict(np.zeros(n, dtype=np.float32))
        self.assertEqual(out.shape, (self.m.n_frames(n), len(self.m.config['classes'])))

    def test_a_short_chunk_is_padded_but_the_padding_is_not_reported(self):
        n = self.m.samples_min + 5
        out = self.m.predict(np.zeros(n, dtype=np.float32))
        self.assertEqual(out.shape[0], self.m.n_frames(n))
        self.assertLess(out.shape[0], self.m.n_frames(self.m.samples_session))

    def test_padding_does_not_change_the_frames_that_are_reported(self):
        # The same audio analyzed alone and analyzed with silence after it has
        # to give the same answer for the frames it covers.
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(3 * self.m.samplerate).astype(np.float32) * 0.1
        short = self.m.predict(audio)
        longer = self.m.predict(np.concatenate([audio, np.zeros(self.m.samplerate, np.float32)]))
        n_common = short.shape[0] - 1  # the last frame of `short` overlaps the join
        np.testing.assert_allclose(short[:n_common], longer[:n_common], atol=1e-5)

    def test_a_chunk_longer_than_the_session_is_an_error_not_a_silent_truncation(self):
        with self.assertRaises(ValueError):
            self.m.predict(np.zeros(self.m.samples_session + 1, dtype=np.float32))

    def test_a_list_is_accepted_as_well_as_an_array(self):
        out = self.m.predict([0.0] * self.m.samples_min)
        self.assertEqual(out.shape[0], 1)

    def test_initialize_refuses_without_a_session_length(self):
        m = model()
        with self.assertRaises(RuntimeError):
            m.initialize()


class TestAgainstTheGraph(unittest.TestCase):
    """n_frames against the number of rows the ONNX graph actually returns.

    Cheap enough to be worth doing directly: a session at a fixed length builds
    in a fraction of a second, and this is the only check that the arithmetic
    still matches the export.
    """

    def rows_from_graph(self, m, n_samples):
        m.samples_session = n_samples
        m.initialize()
        return m.model.run(None, {m.name_in: np.zeros(n_samples, dtype=np.float32)})[0].shape[0]

    def test_matches_the_graph_at_the_lengths_that_matter(self):
        m = model()
        lengths = [
            m.samples_min,                    # exactly one frame
            m.samples_min + 1,                # one sample into the second
            m.samples_min + m.samples_hop,    # exactly two
            61680,                            # the float32 reciprocal case
            16000,                            # a second of audio, under a frame
            5 * 16000,                        # an ordinary short chunk
        ]
        for n in lengths:
            with self.subTest(n_samples=n):
                self.assertEqual(m.n_frames(n), self.rows_from_graph(m, n))


if __name__ == '__main__':
    unittest.main()
