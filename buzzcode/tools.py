import tensorflow as tf
import tensorflow_io as tfio
import os
import re


def loadUp(modelname):
    model = tf.keras.models.load_model(os.path.join("./models/", modelname))

    classes = []
    with open(os.path.join("models/", modelname, "classes.txt"), "r") as file:
        # Use a for loop to read each line from the file and append it to the list
        for line in file:
            # Remove the newline character and append the item to the list
            classes.append(line.strip())
    return model, classes


def load_flac(filepath):
    """ Load a FLAC file, convert it to a float tensor """  # I removed resampling as this should be done in preprocessing
    flac_contents = tf.io.read_file(filepath)
    flac_tensor = tfio.audio.decode_flac(flac_contents, dtype=tf.int16)
    flac_tensor = tf.squeeze(flac_tensor, axis=-1)
    flac_32 = tf.cast(flac_tensor, tf.float32)
    flac_normalized = (flac_32 + 1) / (65536 / 2)  # I think this is the right operation; tf.audio.decode_wav says:
    # "-32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float"
    # dividing by (65536/2) makes the max 1.0 and the min a rounding error from 0

    return flac_normalized


def load_wav(filepath):
    """ Load a WAV file, convert it to a float tensor """  # I removed resampling as this should be done in preprocessing
    wav_contents = tf.io.read_file(filepath)
    wav, sample_rate = tf.audio.decode_wav(
        wav_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav


def load_audio(filepath):
    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".wav":
        data = load_wav(filepath)
    elif extension == ".flac":
        data = load_flac(filepath)
    else:
        quit("buzzdetect only supports .wav and .flac files for analysis")

    return data


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