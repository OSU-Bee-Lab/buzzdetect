import os

import numpy as np
import pandas as pd

from buzzcode.config import SUFFIX_RESULT_COMPLETE, SUFFIX_RESULT_PARTIAL


def melt_coverage(cover_df, framelength=None):
    """ where cover_df is a dataframe with start and end columns OR framelength is provided"""
    if 'end' not in cover_df.columns and framelength is None:
        raise ValueError('cover_df has no "end" column and framelength is not provided')

    cover_df = cover_df.copy()  # don't mutate
    if 'end' not in cover_df.columns:
        cover_df['end'] = cover_df['start'] + framelength

    cover_df.sort_values("start", inplace=True)
    cover_df["coverageGroup"] = (cover_df["start"] > cover_df["end"].shift()).cumsum()
    df_coverage = cover_df.groupby("coverageGroup").agg({"start": "min", "end": "max"})

    coverage = list(zip(df_coverage['start'], df_coverage['end']))

    return coverage  # list of tuples


def get_gaps(range_in, coverage_in):
    coverage_in = sorted(coverage_in)
    gaps = []

    # gap between range start and coverage start
    if coverage_in[0][0] > range_in[0]:  # if the first coverage starts after the range starts
        gaps.append((0, coverage_in[0][0]))

    # gaps between coverages
    for i in range(0, len(coverage_in) - 1):
        out_current = coverage_in[i]
        out_next = coverage_in[i + 1]

        if out_next[0] > out_current[1]:  # if the next coverage starts after this one ends,
            gaps.append((out_current[1], out_next[0]))

    # gap after coverage
    if coverage_in[len(coverage_in) - 1][1] < range_in[1]:  # if the last coverage ends before the range ends
        gaps.append((coverage_in[len(coverage_in) - 1][1], range_in[1]))

    return gaps


def smooth_gaps(gaps, range_in, framelength, gap_tolerance):
    # ignore gaps that start less than 1 frame from range end
    gaps = [gap for gap in gaps if gap[0] < (range_in[1] - framelength)]

    if gap_tolerance is not None:
        gaps = [gap for gap in gaps if (gap[1] - gap[0]) > gap_tolerance]

    # expand from center gaps smaller than one frame; leave gaps larger than one frame
    gaps = [(gap[0] - framelength / 2, gap[0] + framelength / 2) if (gap[1] - gap[0]) < framelength else gap for gap in
            gaps]

    return gaps


def gaps_to_chunklist(gaps_in, chunklength, decimals=2):
    chunklist = []

    for gap in gaps_in:
        chunkpoints = np.arange(gap[0], gap[1], chunklength).tolist()
        chunkpoints.append(gap[1])  # arange is not right-inclusive, even if right perfectly aligns
        chunkpoints = np.round(chunkpoints, decimals)  # floating point errors lead to unintelligible output; 1/100s rounding is fine.
        chunks_in_gap = list(zip(chunkpoints[:-1], chunkpoints[1:]))  # paired tuples of chunkpoints

        chunklist.extend(chunks_in_gap)

    return chunklist
