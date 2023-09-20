import tensorflow as tf

def loadUp(path):
    return tf.keras.models.load_model(path)

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor """ # I removed resampling as this should be done in preprocessing
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav