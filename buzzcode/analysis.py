import json
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from buzzcode.config import SUFFIX_RESULT_COMPLETE, SUFFIX_RESULT_PARTIAL
from buzzcode.utils import search_dir


# Functions for handling models
#

def load_model(modelname):
    dir_model = os.path.join('./models', modelname)
    model = tf.keras.models.load_model(dir_model)

    return model


def load_model_config(modelname):
    dir_model = os.path.join('./models', modelname)
    with open(os.path.join(dir_model, 'config.txt'), 'r') as cfg:
        config = json.load(cfg)

    return config


# Functions for applying transfer model
#

def translate_results(results, classes, digits=2):
    # as I move from DataFrames to dicts, this function is becoming less useful...
    results = np.array(results)
    results = results.round(digits)
    translate = []
    for i, scores in enumerate(results):
        results_frame = {classes[i]: scores[i] for i in range(len(classes))}
        translate.append(results_frame)
    output_df = pd.DataFrame(translate)

    return output_df


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


def get_coverage(path_raw, dir_raw, dir_out, framelength=None, framelength_digits=None):
    raw_base = os.path.splitext(path_raw)[0]

    path_partial = re.sub(dir_raw, dir_out, raw_base) + SUFFIX_RESULT_PARTIAL
    if os.path.exists(path_partial):
        path_out = path_partial
    else:
        path_out = re.sub(dir_raw, dir_out, raw_base) + SUFFIX_RESULT_COMPLETE

    if not os.path.exists(path_out):
        return []

    out_df = pd.read_csv(path_out)
    if 'end' not in out_df.columns:
        if framelength is None:
            raise ValueError('Cannot get coverage; output has no \'end\' column and user did not provide framelength')
        out_df['end'] = round(out_df['start'] + framelength, framelength_digits)

    coverage = melt_coverage(out_df)
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


def chunklist_from_base(base_out, duration_audio, framelength, chunklength):
    path_complete = base_out + SUFFIX_RESULT_COMPLETE

    # if finished analysis file exists, return no chunks
    if os.path.exists(path_complete):
        return []

    path_partial = base_out + SUFFIX_RESULT_PARTIAL

    # if the file hasn't been started, chunk the whole file
    if not os.path.exists(path_partial):
        gaps = [(0, duration_audio)]
    else:
        # otherwise, read the file and calculate chunks
        df = pd.read_csv(path_partial)
        coverage = melt_coverage(df, framelength)
        gaps = get_gaps(
            range_in=(0, duration_audio),
            coverage_in=coverage
        )
        gaps = smooth_gaps(
            gaps,
            range_in=(0, duration_audio),
            framelength=framelength,
            gap_tolerance=framelength / 4
        )

    # if we find no gaps, this file was actually finished!
    # output to the finished file
    if not gaps:
        df.sort_values("start", inplace=True)
        df.to_csv(path_complete, index=False)
        os.remove(path_partial)

        # return no chunks, because the file was finished
        return []

    chunklist = gaps_to_chunklist(gaps, chunklength)
    return chunklist

