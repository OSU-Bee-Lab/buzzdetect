import glob
import json
import multiprocessing
import os
import pickle
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import buzzcode.config as cfg
from buzzcode.analysis.coverage import melt_coverage
from buzzcode.audio import frame_audio, get_duration
from buzzcode.embedding.load_embedder import load_embedder


def overlaps(range_frame, range_label, minimum_overlap):
    range_overlap = (range_frame[0] + minimum_overlap, range_frame[1] - minimum_overlap)

    # if label ends before frame starts or label starts after frame ends, False
    if (range_label[1] < range_overlap[0]) or (range_label[0] > range_overlap[1]):
        return False
    else:
        # otherwise, it must be overlapping, no matter how!
        return True


def frame_events(range_frame, annotations, event_overlap_s):
    overlap_mask = [overlaps(range_frame, (start, end), event_overlap_s) for (start, end) in zip(annotations['start'], annotations['end'])]
    events = annotations['label'][overlap_mask].unique().tolist()
    events = sorted(events)

    return events


def expand_chunk(chunk, framelength, event_overlap_s, audio_duration):
    """ Chunks need to be expanded so that framing starts and ends with the frame overlapping by event_overlap_s"""
    if audio_duration < framelength:
        raise ValueError('input audio is shorter than one frame')

    start = max(
        # nudge start a bit forward, otherwise events might not quite overlap
        chunk[0] - ((framelength - event_overlap_s) * 0.99),
        0  # if start < 0, round up to 0
    )

    end = min(
        chunk[1] + (framelength - event_overlap_s),
        audio_duration  # if end > file length, round down to file length
    )

    # if the annotation is right at the start of the file and less than a frame,
    # it can't be expanded from the center, so expand right. Vice verca for end of file.
    if (end - start) < framelength:
        if end == audio_duration:
            start = end - (framelength*0.99)
        elif start == 0:
            end = start + framelength
    return start, end



def collapse_labels(labellist):
    seperator = '+'
    collapse = seperator.join(labellist)
    return collapse


def get_ident_audio_path(ident):
    base_raw = os.path.join(cfg.TRAIN_DIR_AUDIO, ident)

    path_audio = glob.glob(base_raw + '.*')

    if len(path_audio) > 1:
        raise ValueError(f'multiple audio files found for ident {ident}')
    elif not path_audio:
        return []

    path_audio = path_audio[0]

    return path_audio


def extract_ident(setname, ident, framelength, samplerate_target, framehop_s, overlap_event_s, embedder_dtype):
    print(f'EXTRACTION: building ident {ident}')
    path_audio = get_ident_audio_path(ident)
    if not path_audio:
        warnings.warn(f'audio file not found for {ident}; skipping extraction')
        return None

    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    dir_out = os.path.join(dir_set, cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_RAW, ident)

    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    annotations = annotations[annotations['ident'] == ident].copy()

    track = sf.SoundFile(path_audio)
    sr_native = track.samplerate

    audio_duration = get_duration(path_audio)
    chunks_raw = melt_coverage(annotations)
    chunks = [expand_chunk(chunk, framelength, overlap_event_s, audio_duration) for chunk in chunks_raw]

    frames_by_label = {}

    def process_chunk(chunk):
        start_sample = round(sr_native * chunk[0])
        samples_to_read = round(sr_native * (chunk[1] - chunk[0]))
        track.seek(start_sample)

        audio_data = track.read(samples_to_read, dtype=embedder_dtype)
        if track.channels > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_data = librosa.resample(y=audio_data, orig_sr=sr_native, target_sr=samplerate_target)

        frames = frame_audio(audio_data=audio_data, framelength=framelength, samplerate=samplerate_target, framehop_s=framehop_s)

        frametimes = chunk[0] + (np.arange(0, len(frames)) * framehop_s)
        frametimes = [(s, s + framelength) for s in frametimes]

        for frame, frame_range in zip(frames, frametimes):
            events_frame = frame_events(
                range_frame=frame_range,
                annotations=annotations,
                event_overlap_s=overlap_event_s
            )

            labels_collapse = collapse_labels(events_frame)
            if labels_collapse not in frames_by_label:
                frames_by_label[labels_collapse] = [frame]
            else:
                frames_by_label[labels_collapse].append(frame)

    for chunk in chunks:
        process_chunk(chunk)

    os.makedirs(dir_out, exist_ok=True)
    for labels_collapse, samples in frames_by_label.items():
        path_pickle = os.path.join(dir_out, labels_collapse + '.pickle')
        with open(path_pickle, 'wb') as files:
            for s in samples:
                pickle.dump(s, files)
    return None

def ident_is_extracted(ident, setname):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    dir_out = os.path.join(dir_set, cfg.TRAIN_DIRNAME_AUDIOSAMPLES, cfg.TRAIN_DIRNAME_RAW, ident)

    return bool(os.path.exists(dir_out) and os.listdir(dir_out))

def extract_set(setname, cpus):
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)

    annotations = pd.read_csv(os.path.join(dir_set, 'annotations.csv'))
    idents = annotations['ident'].unique()

    idents_extracted = [i for i in idents if ident_is_extracted(i, setname)]
    idents_remaining = [i for i in idents if i not in idents_extracted]
    idents_remaining = sorted(idents_remaining)

    if not idents_remaining:
        print(f'all idents already extracted')
        return None

    if len(idents_remaining) != len(idents):
        print(
            f'skipping extraction for {len(idents)-len(idents_remaining)} already-extracted idents\n'
            f'NOTE: this will skip the entire ident; new annotations or partially-extracted idents\n'
            f'will NOT be completed. Delete partial ident folders if you wish to re-extract.'
        )

    with open(os.path.join(dir_set, 'config_set.json'), 'r') as file:
        config_set = json.load(file)

    embedder = load_embedder(config_set['embeddername'], framehop_prop=config_set['framehop_prop'], load_model=False)
    framelength = embedder.framelength_s
    samplerate_target = embedder.samplerate
    framehop_s = embedder.framehop_s
    overlap_event_s = config_set['event_overlap_prop']*embedder.framelength_s
    embedder_dtype = embedder.dtype_in

    with multiprocessing.Pool(cpus) as pool:
        # TODO: refactor to OOP
        arglist = [(setname, ident, framelength, samplerate_target, framehop_s, overlap_event_s, embedder_dtype) for ident in idents]
        pool.starmap_async(extract_ident, arglist)
        pool.close()
        pool.join()

    print(f'all extractions complete\n:)\n:D\n:O')
    return True

if __name__ == '__main__':
    extract_set(
        setname='standard',
        cpus=4
    )
