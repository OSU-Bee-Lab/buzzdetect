import glob
import os
import re

import librosa
import numpy as np
import soundfile as sf

from buzzcode.config import tag_eof


def get_duration(path_audio):
    track = sf.SoundFile(path_audio)

    base_eof = os.path.splitext(path_audio)[0] + tag_eof
    paths_eof = glob.glob(base_eof + '*')

    if paths_eof:
        if len(paths_eof) > 1:
            raise ValueError(f"multiple EOF files found for {path_audio}")
        frame_final = int(re.search(base_eof + "_(.*)", paths_eof[0]).group(1))
    else:
        frame_final = track.frames

    return frame_final / track.samplerate


def frame_audio(audio_data, framelength, samplerate, framehop_s):
    framelength_samples = int(framelength * samplerate)
    audio_samples = len(audio_data)
    if audio_samples < framelength_samples:
        raise ValueError('sub-frame audio given')

    step = int(framehop_s * samplerate)

    frames = []
    # yields consecutive audio frames, stopping before exceeding audio length
    for frame_start in range(0, audio_samples - framelength_samples + 1, step):
        frames.append(audio_data[frame_start:frame_start + framelength_samples])

    return frames


def mark_eof(path_audio, frame_final):
    path_eof = os.path.splitext(path_audio)[0] + tag_eof + '_' + str(frame_final)
    open(path_eof, 'a').close()


def stream_to_queue(path_audio, chunklist, q_assignments, resample_rate, mark_bad_eof=True):
    def queue_assignment(chunk, track, samplerate_native):
        sample_from = int(chunk[0] * samplerate_native)
        sample_to = int(chunk[1] * samplerate_native)
        read_size = sample_to - sample_from

        track.seek(sample_from)
        samples = track.read(read_size)
        if track.channels > 1:
            samples = np.mean(samples, axis=1)

        # we've found that many of our .mp3 files give an incorrect .frames count, or else headers are broken
        # this appears to be because our recorders ran out of battery while recording
        # SoundFile does not handle this gracefully, so we catch it here.
        n_samples = len(samples)
        if n_samples < read_size:
            perc = int((n_samples / read_size) * 100)
            f"unexpectedly small read for chunk {chunk} for file {path_audio}. "
            f"Received {perc}% of samples requested ({read_size}/{n_samples})"

        samples = librosa.resample(y=samples, orig_sr=samplerate_native, target_sr=resample_rate)

        assignment = {
            'path_audio': path_audio,
            'chunk': chunk,
            'samples': samples
        }

        q_assignments.put(assignment)

    track = sf.SoundFile(path_audio)
    samplerate_native = track.samplerate

    for chunk in chunklist:
        queue_assignment(chunk, track, samplerate_native)

    audio_duration = track.frames / track.samplerate
    chunk_near_end = abs(chunk[1] - audio_duration) < 10

    if mark_bad_eof and chunk_near_end:
        mark_eof(path_audio, track.tell())

    track.close()
