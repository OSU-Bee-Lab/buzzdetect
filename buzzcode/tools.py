import tensorflow as tf
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

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor """ # I removed resampling as this should be done in preprocessing
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

