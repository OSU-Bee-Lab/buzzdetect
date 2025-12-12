"""
Handling of audio files

Most functions are for custom handling of audio duration.
soundfile.SoundFile.frames is not reliable for determining duration.
The "EOF" (end of file) handling provides a reliable way for buzzdetect to
determine the actual readable duration of audio files.

"""

import glob
import multiprocessing
import os
import re

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

from buzzcode.analysis.assignments import AssignLog
from buzzcode.config import TAG_EOF


def get_duration(path_audio, q_log: multiprocessing.Queue = None):
    """
    Get the duration of an audio file in seconds.

    If EOF files exist, use those to determine the actual duration.
    If multiple EOF files exist, call resolve_multiple_eof().

    Parameters:
    -----------
    path_audio : str
        Path to audio file
    q_log : multiprocessing.Queue or None
        The queue for logging messages. If None, no messages will be logged.

    Returns:
    --------
    float
        The duration of the audio file in seconds.
    """
    track = sf.SoundFile(path_audio)
    paths_eof = enumerate_eof_files(path_audio)

    if not paths_eof:
        frame_final = track.frames
    elif len(paths_eof) == 1:
        frame_final = extract_eof_frame(paths_eof[0])
    else:
        if q_log is not None:
            q_log.put(AssignLog(message=f"multiple EOF files found for {path_audio}; retaining earliest", level_str='WARNING'))
        frame_final = resolve_multiple_eof(paths_eof)

    return frame_final / track.samplerate

def extract_eof_frame(path_eof):
    """
    Extract the final frame number from a path to an EOF file
    """
    matches = re.findall(rf'{TAG_EOF}_(\d+)$', path_eof)
    if not matches:
        raise ValueError(f'Could not extract eof frame from {path_eof}')
    elif len(matches) > 1:
        raise ValueError(f'multiple final frame matches found in {path_eof}')
    return int(matches[0])

def resolve_multiple_eof(paths_eof):
    dict_eof = {p: extract_eof_frame(p) for p in paths_eof}
    list_eof = sorted(dict_eof.items(), key=lambda x: x[1])
    for path, _ in list_eof[1:]:
        os.remove(path)

    return list_eof[0][1]

def enumerate_eof_files(path_audio):
    base_eof = os.path.splitext(path_audio)[0] + TAG_EOF
    paths_eof = glob.glob(base_eof + '*')

    return paths_eof


def frame_audio(audio_data, framelength, samplerate, framehop_s):
    """
    Frame audio data using NumPy's sliding_window_view for efficient processing.
    NOTE! Do not use this to pre-frame audio that will be passed to an embedder.
    Framing should be handled within embedders (it often occurs after spectrogram creation)

    Parameters:
    -----------
    audio_data : array-like
        Input audio data
    framelength : float
        Frame length in seconds
    samplerate : int or float
        Sample rate in Hz
    framehop_s : float
        Hop size between frames in seconds

    Returns:
    --------
    np.ndarray
        2D array where each row is a frame
    """
    audio_data = np.asarray(audio_data)
    framelength_samples = int(framelength * samplerate)
    audio_samples = len(audio_data)

    if audio_samples < framelength_samples:
        raise ValueError('sub-frame audio given')

    step = int(framehop_s * samplerate)

    # Use sliding_window_view for efficient framing
    windowed = sliding_window_view(audio_data, window_shape=framelength_samples)

    # Extract frames at specified hop intervals
    frames = windowed[::step]

    return frames


def mark_eof(path_audio, final_frame):
    path_eof = os.path.splitext(path_audio)[0] + TAG_EOF + '_' + str(final_frame)
    open(path_eof, 'a').close()
