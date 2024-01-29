import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os
import re
import librosa
import pickle
import numpy as np
import soundfile as sf

framelength = 960
framehop = 480


# Functions for handling models
#

def loadup(modelname):
    dir_model = os.path.join("./models/", modelname)
    model = tf.keras.models.load_model(dir_model)

    path_weights = os.path.join(dir_model, "weights.csv")

    df = pd.read_csv(path_weights)
    classes = df['classification']
    return model, list(classes)


def get_yamnet():
    os.environ["TFHUB_CACHE_DIR"]="./yamnet"
    yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

    return yamnet


# Functions for mapping analysis
#

def solve_memory(memory_allot, cpus):
    memory_tf = 0.350  # memory (in GB) required for single tensorflow process
    memorydensity_audio = 3600 / 2.4  # seconds per gigabyte of decoded audio (estimate)

    n_analyzers = min(cpus, (memory_allot/memory_tf).__floor__()) # allow as many workers as you have memory for
    memory_remaining = memory_allot - (memory_tf*n_analyzers)
    memory_perchunk = min(memory_remaining/cpus, 9)  # hard limiting max memory per chunk to 9G because I haven't tested sizes above that (also there's probably not a performance benefit?)

    chunklength = memory_perchunk * memorydensity_audio

    if chunklength<(framelength/1000):
        raise ValueError("memory_allot and cpu combination results in illegally small frames")  # illegally smol

    return chunklength, n_analyzers


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


# Functions for handling audio
#

# could be deprecated if I find a way to read filepaths from tensors or rewrite training
def load_audio_tf(path_audio):
    """ Load a WAV file, convert it to a float tensor """
    file_contents = tf.io.read_file(path_audio)
    data, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    data = tf.squeeze(data, axis=-1)
    return data


def load_audio(path_audio, time_start=0, time_stop=None):
    track = sf.SoundFile(path_audio)

    can_seek = track.seekable() # True
    if not can_seek:
        raise ValueError("Input file not compatible with seeking")

    if time_stop is None:
        time_stop = librosa.get_duration(path=path_audio)

    sr = track.samplerate
    start_frame = round(sr * time_start)
    frames_to_read = round(sr * (time_stop - time_start))
    track.seek(start_frame)
    audio_section = track.read(frames_to_read)
    audio_section = librosa.resample(y=audio_section, orig_sr=sr, target_sr=16000)  # overwrite for memory purposes

    return audio_section


def extract_embeddings(audio_data, yamnet, pad=False):
    if audio_data.dtype != 'tf.float32':
        audio_data = tf.cast(audio_data, tf.float32)

    audio_data_split = tf.signal.frame(audio_data, framelength * 16, framehop * 16, pad_end=pad, pad_value=0)

    embeddings = [yamnet(data)[1] for data in audio_data_split]

    return embeddings


# not yet implemented for full-length analysis; won't work for long files (too large for mem)
def save_embeddings(path_pickle, embeddings):
    os.makedirs(os.path.dirname(path_pickle), exist_ok=True)

    with open(path_pickle, 'wb') as file_pickle:
        pickle.dump(embeddings, file_pickle)


def load_embeddings(path_pickle):
    with open(path_pickle, 'rb') as file_pickle:
        embeddings = pickle.load(file_pickle)

    return embeddings


def analyze_embeddings(model, classes, embeddings):
    results = []

    for i, e in enumerate(embeddings):
        scores = model(e).numpy()[0]

        results_frame = {
            "start": (i * framehop) / 1000,
            "end": ((i * framehop) + framelength) / 1000,
            "class_predicted": classes[scores.argmax()],
            "score_predicted": scores[scores.argmax()]
        }

        indices_out = [classes.index(c) for c in classes]
        scorenames = ['score_' + c for c in classes]
        results_frame.update({scorenames[i]: scores[i] for i in indices_out})

        results.append(results_frame)

    output_df = pd.DataFrame(results)

    return output_df
