import glob
import json
import multiprocessing
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import buzzcode.config as cfg
from buzzcode.analysis.coverage import melt_coverage
from buzzcode.audio import frame_audio
from buzzcode.utils import setthreads

setthreads(1)


def extract_ident(ident, annotations_ident, framelength, overlap_event_s, framehop_s, samplerate_target):
    print(f'EXTRACTION: starting ident {ident}')
    base_raw = os.path.join(cfg.DIR_TRAIN_AUDIO, ident + '.*')
    path_raw = glob.glob(base_raw)
    if len(path_raw) > 1:
        raise ValueError(f'multiple audio files found for ident {ident}')
    path_raw = path_raw[0]

    ranges = list(zip(annotations_ident['start'], annotations_ident['end']))

    track = sf.SoundFile(path_raw)
    sr_native = track.samplerate
    audio_duration = librosa.get_duration(path=path_raw)

    def process_chunk(chunk):
        print(f'EXTRACTION: {ident}: starting chunk {(round(chunk[0], 1), round(chunk[1], 1))}')

        start_sample = round(sr_native * chunk[0])
        samples_to_read = round(sr_native * (chunk[1] - chunk[0]))
        track.seek(start_sample)

        audio_data = track.read(samples_to_read, dtype='float32')
        audio_data = librosa.resample(y=audio_data, orig_sr=sr_native, target_sr=samplerate_target)

        frames = frame_audio(audio_data=audio_data, framelength=framelength, samplerate=samplerate_target,
                             framehop_s=framehop_s)

        frametimes = chunk[0] + (np.arange(0, len(frames)) * framehop_s)
        frametimes = [(s, s + framelength) for s in frametimes]
        events_chunk = []

        for frame_range in frametimes:
            events_frame = overlapping_elements(
                range_index=frame_range,
                range_table=ranges,
                elements=annotations_ident['classification'],
                minimum_overlap=overlap_event_s
            ).unique().tolist()

            events_chunk.append(events_frame)

        samples_chunk = [{'audio_data': f, 'labels': e} for f, e in zip(frames, events_chunk)]

        return samples_chunk

    chunks_raw = melt_coverage(annotations_ident)
    chunks_expand = [
        expand_chunk(chunk, framelength, overlap_event_s, audio_duration) for
        chunk in chunks_raw]

    samples_ident = []
    for chunk in chunks_expand:
        samples_ident.extend(process_chunk(chunk))

    return samples_ident


def collapse_labels(labellist):
    seperator = '+'
    collapse = seperator.join(labellist)
    return collapse


def write_ident(setname, ident, framehop_s, framelength, samplerate_target, overlap_event_s):
    dir_set = os.path.join(cfg.DIR_TRAIN_SET, setname)
    dir_ident = os.path.join(dir_set, 'samples_audio', ident)
    os.makedirs(dir_ident, exist_ok=False)

    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    annotations = annotations[annotations['ident'] == ident].copy()

    samples = extract_ident(
        ident=ident,
        annotations_ident=annotations,
        framehop_s=framehop_s,
        framelength=framelength,
        samplerate_target=samplerate_target,
        overlap_event_s=overlap_event_s
    )

    labels_all = [s['labels'] for s in samples]
    labels_unique = [list(x) for x in set(tuple(x) for x in labels_all)]

    for labels in labels_unique:
        labels_collapse = collapse_labels(labels)

        path_l = os.path.join(dir_ident, labels_collapse + '.pickle')
        samples_l = [s['audio_data'] for s in samples if s['labels'] == labels]

        with open(path_l, 'wb') as files:
            for s in samples_l:
                pickle.dump(s, files)


def extract_set(setname, event_overlap_prop, framehop_s, cpus):
    dir_set = os.path.join(cfg.DIR_TRAIN_SET, setname)

    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    idents = annotations['ident'].unique()

    with open(os.path.join(dir_set, 'config_embedder.txt'), 'r') as f:
        config_embedder = json.load(f)

    framelength = config_embedder['framelength']
    overlap_event_s = framelength * event_overlap_prop
    samplerate_target = config_embedder['samplerate']

    def build_args(ident):
        args = (
            setname,
            ident,
            framehop_s,
            framelength,
            samplerate_target,
            overlap_event_s
        )

        return args

    arglist = [build_args(ident) for ident in idents]
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.starmap_async(write_ident, arglist)  # have to starmap cuz can't pickle local function
        pool.close()
        pool.join()


def overlapping_elements(range_index, range_table, minimum_overlap, elements=None):
    """given an input range to index with, what elements of the table (list) match?"""
    ''' if elements is provided, returns matching elements. Else returns a boolean list of matches'''

    overlap_bool = [overlaps(range_index, r_table, minimum_overlap) for r_table in range_table]
    if elements is None:
        return overlap_bool

    return elements[overlap_bool]


def overlaps(range_frame, range_label, minimum_overlap):
    range_overlap = (range_frame[0] + minimum_overlap, range_frame[1] - minimum_overlap)

    # if label ends before frame starts or label starts after frame ends, False
    if (range_label[1] < range_overlap[0]) or (range_label[0] > range_overlap[1]):
        return False
    else:
        # otherwise, it must be overlapping, no matter how!
        return True


def expand_chunk(chunk, framelength, event_overlap_s, audio_duration):
    start = max(
        chunk[0] - ((framelength - event_overlap_s) * 0.99),
        # nudge start a bit forward, rounding can lead to accidental nulls where real overlap is *just* less than tolerable
        0  # if start < 0, round up to 0
    )

    end = min(
        chunk[1] + (framelength - event_overlap_s),
        audio_duration  # if end > file length, round down to file length
    )

    if (end - start) < framelength:
        print(f'unexpandable sub-frame audio at; returning null')
        return

    return start, end
