import json
import os

import numpy as np
import tensorflow as tf

import buzzcode.config as cfg
from buzzcode.utils import setthreads

# TODO: can we pull YAMNet from tf_hub? It seems to have some issues

setthreads(1)


def load_yamnet(framehop_s):
    """Create a YAMNet model with specified frame hop."""
    dir_yamnet = os.path.join(cfg.dir_embedders, 'yamnet')

    model = tf.keras.models.load_model(dir_yamnet, compile=False)
    modelconf = model.get_config()
    framehop_base = modelconf['layers'][20]['inbound_nodes'][0][3]['frame_step'] / 100
    if framehop_s == framehop_base:
        return model

    # Convert seconds to centiseconds for TF config
    framehop_cs = int(framehop_s * 100)
    modelconf['layers'][20]['inbound_nodes'][0][3]['frame_step'] = framehop_cs

    model_rehop = tf.keras.Model.from_config(modelconf)
    model_rehop.set_weights(model.get_weights())
    return model_rehop


def config_yamnet():
    dir_yamnet = os.path.join(cfg.dir_embedders, 'yamnet')
    with open(os.path.join(dir_yamnet, 'config.txt')) as f:
        config = json.load(f)
    return config


def load_embedder_model(embeddername, framehop_s):
    if embeddername.lower() == 'yamnet':
        return load_yamnet(framehop_s)

    # TODO: make work!
    elif embeddername.lower() == 'birdnet':
        raise ValueError('embedding with birdnet is currently broke :(')
        from tensorflow import lite as tflite

        path_model = 'embedders/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'

        birdnet = tflite.Interpreter(model_path=path_model, num_threads=1)
        birdnet.allocate_tensors()

        # Get input and output tensors.
        input_details = birdnet.get_input_details()
        output_details = birdnet.get_output_details()

        # Get input tensor index
        input_index = input_details[0]["index"]

        #  drops output layer, returning embeddings instead of classifications
        output_index = output_details[0]["index"] - 1

        config = dict(
            embeddername='birdnet',
            framelength=3,
            samplerate=48000,
            n_embeddings=1024
        )

        def embedder(frames):
            """
            :param frames: a list where each element is a numpy array of audio samples
            :return: a list of equal length to the input where each element is a numpy array of embedding values
            """

            # Reshape input tensor
            birdnet.resize_tensor_input(input_index, [len(frames), *frames[0].shape])
            birdnet.allocate_tensors()

            # Extract feature embeddings
            birdnet.set_tensor(input_index, np.array(frames, dtype="float32"))
            birdnet.invoke()
            embeddings = birdnet.get_tensor(output_index)
            embeddings = list(embeddings)

            return embeddings

    else:
        print('ERROR: invalid embedder name')
        return

    return embedder


def load_embedder_config(embeddername):
    if embeddername.lower() == 'yamnet':
        return config_yamnet()
