import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import os


def loadup(modelname):
    model = tf.keras.models.load_model(os.path.join("./models/", modelname))

    classes = []
    with open(os.path.join("models/", modelname, "classes.txt"), "r") as file:
        # Use a for loop to read each line from the file and append it to the list
        for line in file:
            # Remove the newline character and append the item to the list
            classes.append(line.strip())
    return model, classes

def get_yamnet():
    os.environ["TFHUB_CACHE_DIR"]="./yamnet"
    yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

    return yamnet


def load_flac(filepath):
    """ Load a FLAC file, convert it to a float tensor """
    flac_contents = tf.io.read_file(filepath)
    flac_tensor = tfio.audio.decode_flac(flac_contents, dtype=tf.int16)
    flac_tensor = tf.squeeze(flac_tensor, axis=-1)
    flac_32 = tf.cast(flac_tensor, tf.float32)
    flac_normalized = (flac_32 + 1) / (65536 / 2)  # I think this is the right operation; tf.audio.decode_wav says:
    # "-32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float"
    # dividing by (65536/2) makes the max 1.0 and the min a rounding error from 0

    return flac_normalized


def load_wav(filepath):
    """ Load a WAV file, convert it to a float tensor """
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