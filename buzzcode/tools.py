import os
import re
import argparse
import librosa
import soundfile as sf

# given a list of paths, create the directories necessary to hold the files
def unique_dirs(paths, make=False):
    path_dirs = []
    for path in paths:
        path_dir = os.path.dirname(path)
        path_dirs.append(path_dir)

    dirs_unique = list(set(path_dirs))

    if make == True:
        for path_dir in dirs_unique:
            os.makedirs(path_dir, exist_ok=True)

    return dirs_unique


def size_to_runtime(size_GB, kbps=256):
    runtime = (
            size_GB *  # gigabytes
            8 *  # convert to gigabits
            (10 ** 6) *  # convert to kilobits
            (1 / kbps)
    )  # convert to seconds

    return runtime


def runtime_to_size(runtime, kbps=256):
    size_GB = (
            runtime *  # seconds
            kbps *  # kilobits
            (1 / 8) *  # kilobytes
            (10 ** -6)  # gigabytes
    )

    return size_GB


def clip_name(filepath, clippath):
    clip = re.sub(clippath, "", filepath)
    return clip


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def search_dir(dir_in, extensions):
    if not (extensions.__class__ is list and extensions[0].__class__ is str):
        raise ValueError("input extensions should be list of strings")

    # convert extensions into regex, if they aren't already
    for i in range(len(extensions)):
        e = extensions[i]
        if e[len(e) - 1] != "$":
            e = e+"$"

        e = e.lower()
        extensions[i] = e

    paths = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            file = file.lower()

            if True in [bool(re.search(e, file)) for e in extensions]:
                paths.append(os.path.join(root, file))

    return paths

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