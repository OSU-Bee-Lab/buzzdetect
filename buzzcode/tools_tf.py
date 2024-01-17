import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os


def loadup(modelname):
    dir_model = os.path.join("./models/", modelname)
    model = tf.keras.models.load_model(dir_model)

    classpath_txt =os.path.join(dir_model, "classes.txt") # backwards compatibility for previous models; can remove later
    classpath_csv = os.path.join(dir_model, "weights.csv")

    if os.path.exists(classpath_csv): # prefer weights if available
        df = pd.read_csv(classpath_csv)
        classes = df['classification']
        return model, list(classes)

    elif os.path.exists(classpath_txt):
        classes = []
        with open(os.path.join("models/", modelname, "classes.txt"), "r") as file:
            # Use a for loop to read each line from the file and append it to the list
            for line in file:
                # Remove the newline character and append the item to the list
                classes.append(line.strip())
        return model, classes
    else:
        raise OSError("no file containing classes found in model directory")

def get_yamnet():
    os.environ["TFHUB_CACHE_DIR"]="./yamnet"
    yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

    return yamnet

# loads audio from a filepath, including a filepath held in a tensor
# could be deprecated if I find a way to read filepaths from tensors; but apparently this is difficult
def load_audio_tf(path_audio):
    """ Load a WAV file, convert it to a float tensor """
    file_contents = tf.io.read_file(path_audio)
    data, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    data = tf.squeeze(data, axis=-1)
    return data
