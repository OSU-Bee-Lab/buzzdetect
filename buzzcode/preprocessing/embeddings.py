import tensorflow_hub as hub
import numpy as np
import os


def get_embedder(embeddername):
    if embeddername.lower() == 'yamnet':
        os.environ["TFHUB_CACHE_DIR"]="./embedders/yamnet"
        yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

        def embedder(data):
            embeddings = yamnet(data)[1]  # element 1 is embeddings
            embeddings = np.array(embeddings)[0, ]  # numpy converts to (1024,1) by default
            return embeddings

        config = dict(
            embeddername='yamnet',
            framelength=0.96,
            samplerate=16000,
            n_embeddings=1024
        )

    elif embeddername.lower() == 'birdnet':
        import buzzcode.BirdNET as bn

        embedder = bn.model.embeddings()

        config = dict(
            embeddername='birdnet',
            framelength=3,
            samplerate=48000,
            n_embeddings=1024
        )

    else:
        print('ERROR: invalid embedder name')
        return

    return embedder, config


def extract_embeddings(frame_data, embedder):
    embeddings = embedder(frame_data)
    return embeddings
