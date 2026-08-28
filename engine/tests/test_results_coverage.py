"""src/stream/results_coverage.py: resuming a partly-analyzed file.

This is what decides which seconds of a file a resumed run re-reads. Getting it
wrong either loses audio (a gap never re-analyzed) or duplicates it, so the
edges matter more than the happy path.
"""

import unittest

import pandas as pd

import tests._context as ctx  # noqa: F401

from src.stream.results_coverage import (gaps_to_chunklist, get_gaps,
                                         melt_coverage, smooth_gaps)

FRAME = 0.96


def frame_starts(*starts):
    return pd.DataFrame({'start': list(starts)})


class TestMeltCoverage(unittest.TestCase):
    def test_contiguous_frames_melt_into_one_span(self):
        df = frame_starts(0.0, 0.96, 1.92)
        self.assertEqual(melt_coverage(df, FRAME), [(0.0, 2.88)])

    def test_a_hole_splits_the_span(self):
        df = frame_starts(0.0, 0.96, 10.0)
        self.assertEqual(melt_coverage(df, FRAME), [(0.0, 1.92), (10.0, 10.96)])

    def test_unsorted_input_is_handled(self):
        df = frame_starts(1.92, 0.0, 0.96)
        self.assertEqual(melt_coverage(df, FRAME), [(0.0, 2.88)])

    def test_explicit_end_column_is_used(self):
        df = pd.DataFrame({'start': [0.0, 5.0], 'end': [2.0, 7.0]})
        self.assertEqual(melt_coverage(df), [(0.0, 2.0), (5.0, 7.0)])

    def test_needs_an_end_or_a_framelength(self):
        with self.assertRaises(ValueError):
            melt_coverage(frame_starts(0.0))

    def test_input_frame_is_not_mutated(self):
        df = frame_starts(0.0, 0.96)
        melt_coverage(df, FRAME)
        self.assertEqual(list(df.columns), ['start'])


class TestGetGaps(unittest.TestCase):
    def test_full_coverage_leaves_no_gap(self):
        self.assertEqual(get_gaps((0, 10), [(0, 10)]), [])

    def test_gap_before_after_and_between(self):
        gaps = get_gaps((0, 30), [(5, 10), (20, 25)])
        self.assertEqual(gaps, [(0, 5), (10, 20), (25, 30)])

    def test_unsorted_coverage_is_sorted_first(self):
        self.assertEqual(get_gaps((0, 30), [(20, 25), (5, 10)]), [(0, 5), (10, 20), (25, 30)])

    def test_range_not_starting_at_zero(self):
        # The leading gap runs from the range's own start, not from zero.
        self.assertEqual(get_gaps((10, 30), [(20, 25)]), [(10, 20), (25, 30)])

    def test_empty_coverage_is_one_gap_over_the_whole_range(self):
        self.assertEqual(get_gaps((0, 30), []), [(0, 30)])


class TestSmoothGaps(unittest.TestCase):
    def test_gap_within_a_frame_of_the_end_is_dropped(self):
        gaps = [(29.5, 30.0)]
        self.assertEqual(smooth_gaps(gaps, (0, 30), FRAME, None), [])

    def test_gap_below_tolerance_is_dropped(self):
        gaps = [(5.0, 5.1)]
        self.assertEqual(smooth_gaps(gaps, (0, 30), FRAME, gap_tolerance=0.24), [])

    def test_sub_frame_gap_is_widened_around_its_start(self):
        gaps = smooth_gaps([(5.0, 5.5)], (0, 30), FRAME, None)
        self.assertEqual(gaps, [(5.0 - FRAME / 2, 5.0 + FRAME / 2)])

    def test_gap_longer_than_a_frame_is_left_alone(self):
        self.assertEqual(smooth_gaps([(5.0, 12.0)], (0, 30), FRAME, None), [(5.0, 12.0)])

    def test_no_tolerance_keeps_small_gaps(self):
        self.assertEqual(len(smooth_gaps([(5.0, 5.01)], (0, 30), FRAME, None)), 1)


class TestGapsToChunklist(unittest.TestCase):
    def test_a_gap_shorter_than_a_chunk_is_one_chunk(self):
        self.assertEqual(gaps_to_chunklist([(0, 30)], 200), [(0.0, 30.0)])

    def test_a_gap_splits_at_chunk_boundaries_with_a_short_tail(self):
        self.assertEqual(gaps_to_chunklist([(0, 250)], 100),
                         [(0.0, 100.0), (100.0, 200.0), (200.0, 250.0)])

    def test_an_exact_multiple_does_not_produce_an_empty_tail_chunk(self):
        self.assertEqual(gaps_to_chunklist([(0, 200)], 100), [(0.0, 100.0), (100.0, 200.0)])

    def test_chunks_are_contiguous_and_cover_the_gap_exactly(self):
        chunks = gaps_to_chunklist([(3.3, 77.7)], 10)
        self.assertEqual(chunks[0][0], 3.3)
        self.assertEqual(chunks[-1][1], 77.7)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(a[1], b[0])

    def test_every_gap_is_chunked(self):
        chunks = gaps_to_chunklist([(0, 50), (100, 150)], 200)
        self.assertEqual(chunks, [(0.0, 50.0), (100.0, 150.0)])

    def test_offsets_are_rounded_to_the_documented_precision(self):
        chunks = gaps_to_chunklist([(0, 1 / 3)], 10)
        self.assertEqual(chunks, [(0.0, 0.33)])


class TestResumeEndToEnd(unittest.TestCase):
    """The sequence WorkerStreamer._chunk_file actually runs."""

    def test_partial_results_produce_chunks_covering_only_the_missing_audio(self):
        duration = 100.0
        # An earlier run got through the first 20 seconds.
        df = frame_starts(*[round(i * FRAME, 2) for i in range(21)])
        coverage = melt_coverage(df, FRAME)
        gaps = get_gaps((0, duration), coverage)
        gaps = smooth_gaps(gaps, (0, duration), FRAME, FRAME / 4)
        chunks = gaps_to_chunklist(gaps, 50)
        self.assertEqual(chunks[0][0], round(20 * FRAME + FRAME, 2))
        self.assertEqual(chunks[-1][1], duration)

    def test_a_fully_covered_file_produces_no_chunks(self):
        duration = 20.16  # exactly 21 frames
        df = frame_starts(*[round(i * FRAME, 2) for i in range(21)])
        gaps = get_gaps((0, duration), melt_coverage(df, FRAME))
        gaps = smooth_gaps(gaps, (0, duration), FRAME, FRAME / 4)
        self.assertEqual(gaps_to_chunklist(gaps, 50), [])


if __name__ == '__main__':
    unittest.main()
