import os

from buzzcode.embedding.BaseEmbedder import BaseEmbedder

class EmbedderYamnetK2(BaseEmbedder):
    # Class attributes - no config file needed!
    embeddername = "yamnet"
    framelength_s = 0.96  # seconds
    digits_time = 2
    samplerate = 16000  # Hz
    n_embeddings = 1024
    dtype_in = 'float32'

    def load(self):
        from keras.layers import TFSMLayer
        if self.framehop_prop == 1:
            dir_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/yamnet_wholehop')
        elif self.framehop_prop == 0.5:
            dir_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/yamnet_halfhop')
        else:
            raise ValueError('For Keras 2 YAMNet, framehop_prop must be 1 or 0.5')
            # TODO: hop on command; totally worth it with how much faster k2 is

        return TFSMLayer(dir_model, call_endpoint='serving_default')


    def embed(self, audio):
        """
        Generate embeddings for audio data

        Args:
            audio: numpy array of audio samples at self.samplerate

        Returns:
            numpy array of embeddings
        """
        return self.model(audio)['global_average_pooling2d']