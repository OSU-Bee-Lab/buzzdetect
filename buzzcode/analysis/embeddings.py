import os

import tensorflow as tf
import tensorflow_hub as hub


def get_embedder(embeddername):
    if embeddername.lower() == 'yamnet':
        os.environ["TFHUB_CACHE_DIR"]="./embedders/yamnet"
        yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

        def embedder(data):
            return yamnet(data)[1]

        framelength = 0.96
        samplerate = 16000

    elif embeddername.lower() == 'birdnet':
        import buzzcode.BirdNET as bn

        embedder= bn.model.embeddings()
        framelength = 3
        samplerate = 48000
    else:
        print('ERROR: invalid embedder name')

    return embedder, framelength, samplerate


def extract_embeddings(audio_data, embedder, pad=False):
    embedder, framelength, samplerate = get_embedder()

    if audio_data.dtype != 'tf.float32':
        audio_data = tf.cast(audio_data, tf.float32)

    audio_data_split = tf.signal.frame(audio_data, framelength * 16, framehop * 16, pad_end=pad, pad_value=0)

    embeddings = [embedder(data)[1] for data in audio_data_split]

    return embeddings
