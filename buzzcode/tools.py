import tensorflow as tf
import tensorflow_io as tfio
import os

def loadUp(modelname):
    model = tf.keras.models.load_model(os.path.join("./models/", modelname))

    classes = []
    with open(os.path.join("models/", modelname, "classes.txt"), "r") as file:
        # Use a for loop to read each line from the file and append it to the list
        for line in file:
            # Remove the newline character and append the item to the list
            classes.append(line.strip())
    return model, classes

def load_flac(flac_path):
    """ Load a FLAC file, convert it to a float tensor """ # I removed resampling as this should be done in preprocessing
    flac_contents = tf.io.read_file(flac_path)
    flac = tfio.audio.decode_flac(flac_contents, dtype=tf.int16)
    return flac



def load_wav_16k_mono(wav_path):
    """ Load a WAV file, convert it to a float tensor """ # I removed resampling as this should be done in preprocessing
    wav_contents = tf.io.read_file(wav_path)
    wav, sample_rate = tf.audio.decode_wav(
          wav_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

# given a list of paths, create the directories necessary to hold the files
def get_unique_dirs(paths, make=True):
    path_dirs = []
    for path in paths:
        path_dir = os.path.dirname(path)
        path_dirs.append(path_dir)

    unique_dirs = list(set(path_dirs))

    if make == True:
        for path_dir in unique_dirs:
            os.makedirs(path_dir, exist_ok=True)

    return unique_dirs
