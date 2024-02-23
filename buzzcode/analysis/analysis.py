import json
import tensorflow as tf
import pandas as pd
import os
import re
import numpy as np


# Functions for handling models
#

def loadup(modelname):
    dir_model = os.path.join("./models/", modelname)
    model = tf.keras.models.load_model(dir_model)

    with open(os.path.join(dir_model, 'config.txt'), 'r') as cfg:
        config = json.load(cfg)

    return model, config


# Functions for mapping analysis
#
def solve_memory(memory_allot, cpus, framelength):
    memory_perproc = memory_allot/cpus
    memory_tf = 0.350  # memory (in GB) required for single tensorflow process

    surplus_perproc = memory_perproc - memory_tf
    memorydensity_audio = 2.4/3600  # gigabytes of memory per second of decoded audio (estimate)

    memory_perframe = framelength*memorydensity_audio

    chunklength = surplus_perproc/memorydensity_audio

    if chunklength < framelength:
        mem_needed = ((cpus * (memory_tf + memory_perframe))*10).__ceil__()/10

        p = ''
        if cpus > 1:
            p = 's'
        raise ValueError(f"too little memory alloted for CPU count; need at least {mem_needed}GB for {cpus} CPU{p}")

    return chunklength


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


def gaps_to_chunklist(gaps_in, chunklength):
    if chunklength < 0.960:
        raise ValueError("chunklength is illegally small")

    chunklist_master = []

    for gap in gaps_in:
        gap_length = gap[1] - gap[0]
        n_chunks = (gap_length/chunklength).__ceil__()
        chunkpoints = np.arange(gap[0], gap[1], chunklength).tolist()

        chunklist_gap = []
        # can probably list comprehend this
        for i in range(len(chunkpoints)-1):
            chunklist_gap.append((chunkpoints[i], chunkpoints[i+1]))

        chunklist_gap.append((chunkpoints[len(chunkpoints) - 1], gap[1]))
        chunklist_master.extend(chunklist_gap)

    return chunklist_master


def get_coverage(path_raw, dir_raw, dir_out):
    out_suffix = "_buzzdetect.csv"

    raw_base = os.path.splitext(path_raw)[0]
    path_out = re.sub(dir_raw, dir_out, raw_base) + out_suffix

    if not os.path.exists(path_out):
        return []

    out_df = pd.read_csv(path_out)
    out_df.sort_values("start", inplace=True)
    out_df["coverageGroup"] = (out_df["start"] > out_df["end"].shift()).cumsum()
    df_coverage = out_df.groupby("coverageGroup").agg({"start": "min", "end": "max"})

    coverage = list(zip(df_coverage['start'], df_coverage['end']))

    return coverage # list of tuples


# Functions for applying transfer model
#

def translate_results(results, classes, framelength):
    scorenames = ['score_' + c for c in classes]

    translate = []
    for i, scores in enumerate(results):
        results_frame = {
            "start": i * (framelength/2),
            "end": ((i * (framelength/2)) + framelength),
            "class_predicted": classes[scores.argmax()],
            "score_predicted": scores[scores.argmax()]
        }

        results_frame.update({scorenames[i]: scores[i] for i in range(len(classes))})

        translate.append(results_frame)

    output_df = pd.DataFrame(translate)

    return output_df



def analyze_input_old(model, classes, framelength, inputs):
    results = []

    framehop = framelength/2

    indices_out = [classes.index(c) for c in classes]
    scorenames = ['score_' + c for c in classes]

    for i, e in enumerate(inputs):
        scores = model(e).numpy()[0]

        results_frame = {
            "start": i * framehop,
            "end": ((i * framehop) + framelength),
            "class_predicted": classes[scores.argmax()],
            "score_predicted": scores[scores.argmax()]
        }

        results_frame.update({scorenames[i]: scores[i] for i in indices_out})

        results.append(results_frame)

    output_df = pd.DataFrame(results)

    return output_df

