"""
Handling of audio files

soundfile.SoundFile.frames is not always reliable for determining duration
(e.g. mp3s from recorders that died mid-recording over-report their length).
The duration here is a best-effort estimate; the streamer catches the true
end of readable audio on-the-fly when a chunk read comes up short.
"""

import multiprocessing

import soundfile as sf

from src.pipeline.assignments import AssignFile


def get_duration(a_file: AssignFile, q_log: multiprocessing.Queue = None):
    """
    Get the duration of an audio file in seconds.

    Parameters:
    -----------
    a_file : AssignFile
        The file to measure. Its track is opened if not already.
    q_log : multiprocessing.Queue or None
        Unused; retained for call-site compatibility.

    Returns:
    --------
    float
        The duration of the audio file in seconds.
    """
    if a_file.track is None:
        a_file.track = sf.SoundFile(a_file.path_audio)

    return a_file.track.frames / a_file.track.samplerate
