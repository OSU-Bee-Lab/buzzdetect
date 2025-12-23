import os

import tensorflow as tf
from tensorflow.keras import Model

from src.embedding.BaseEmbedder import BaseEmbedder
from embedders.yamnet.yamnet import WaveformFeatures  # NEEDED for model loading; custom class in model

"""Feature computation for YAMNet."""

class EmbedderYamnet(BaseEmbedder):
    # Class attributes - no config file needed!
    embeddername = "yamnet"
    framelength_s = 0.96  # seconds
    digits_time = 2
    samplerate = 16000  # Hz
    n_embeddings = 1024
    dtype_in = 'float32'

    def initialize(self):
        """Load the YAMNet model from TensorFlow Hub"""
        curdir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(curdir, 'yamnet.keras')
        model = tf.keras.models.load_model(model_path, compile=False)
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        return model


    def embed(self, audio):
        """
        Generate embeddings for audio data

        Args:
            audio: numpy array of audio samples at self.samplerate

        Returns:
            numpy array of embeddings
        """
        return self.model(audio)