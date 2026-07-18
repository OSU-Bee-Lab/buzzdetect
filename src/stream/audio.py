"""
Handling of audio files

soundfile.SoundFile.frames is not always reliable for determining duration
(e.g. mp3s from recorders that died mid-recording over-report their length).
The duration here is a best-effort estimate; the streamer catches the true
end of readable audio on-the-fly when a chunk read comes up short.
"""

import multiprocessing
import os
import importlib

import soundfile as sf
import src.config as cfg
from src.stream.driver import AudioDriver
from src.pipeline.assignments import AssignFile
from src.utils import get_ext

module_drivers = cfg.DIR_DRIVERS.replace('/', '.')

driver_map = {}

for k in sf.available_formats().keys():
    driver_map[k.lower()] = sf.SoundFile

if os.path.exists(cfg.DIR_DRIVERS):
    for d in os.listdir(cfg.DIR_DRIVERS):
        if not d.endswith('.py'):
            continue

        ext = os.path.splitext(d)[0].lower()
        driver = importlib.import_module(f'{module_drivers}.{ext}').Driver
        driver_map.update({ext: driver})

class UnsupportedFormat(Exception):
    pass

def build_track(path_audio) -> AudioDriver:
    ext = get_ext(path_audio)
    if ext not in driver_map:
        # in theory, this never fires because buzzdetect only looks for supported formats
        raise UnsupportedFormat(f'Unsupported audio format: {ext}')
    return driver_map[ext](path_audio)

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
        a_file.track = build_track(a_file.path_audio)

    return a_file.track.frames / a_file.track.samplerate
