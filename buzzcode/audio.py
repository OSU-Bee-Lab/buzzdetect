import glob
import os
import re
import warnings

import numpy as np
import soundfile as sf

from buzzcode.config import TAG_EOF
from numpy.lib.stride_tricks import sliding_window_view

def get_duration(path_audio):
    track = sf.SoundFile(path_audio)
    base_eof = os.path.splitext(path_audio)[0] + TAG_EOF
    paths_eof = glob.glob(base_eof + '*')

    if not paths_eof:
        frame_final = track.frames
    elif len(paths_eof) == 1:
        match = re.search(base_eof + "_(.*)", paths_eof[0])
        frame_final = int(match.group(1)) if match else track.frames
    else:
        warnings.warn(f"multiple EOF files found for {path_audio}; deleting all")
        for p in paths_eof:
            os.remove(p)
        frame_final = track.frames

    return frame_final / track.samplerate


def frame_audio(audio_data, framelength, samplerate, framehop_s):
    """
    Frame audio data using NumPy's sliding_window_view for efficient processing.

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


def mark_eof(path_audio, frame_final):
    path_eof = os.path.splitext(path_audio)[0] + TAG_EOF + '_' + str(frame_final)
    open(path_eof, 'a').close()


def handle_badread(path_audio, duration_audio, track, end_intended):
    frame_actual = track.tell()

    # if we get a bad read in the middle of a file, this deserves a warning.
    near_end = abs(end_intended - duration_audio) < 20  # not sure there's an objective way to tune this
    if not near_end:
        warnings.warn(f"Unreadable frames starting at {round(frame_actual / track.samplerate, 1)}s for {path_audio}.")
        mark_eof(path_audio, frame_actual)
        return

    # if we get a bad read at the end of a file, this is pretty common when batteries run out.
    # don't bother warning, just mark and move on.
    mark_eof(path_audio, frame_actual)
    return


