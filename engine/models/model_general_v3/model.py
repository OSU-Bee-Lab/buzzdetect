import os

import src.config as cfg
from src.inference.models import BaseModel

class ModelGeneralV3(BaseModel):
    modelname = "model_general_v3"
    embeddername = 'yamnet_k2'
    digits_results = 2

    def initialize(self):
        self.embedder.initialize()

        from keras.layers import TFSMLayer
        dir_model = os.path.abspath(os.path.join(cfg.DIR_MODELS, self.modelname))
        self.model = TFSMLayer(dir_model, call_endpoint='serving_default')  # self.model defined in ABC

    def predict(self, audiosamples):
        """
        Generate predictions for audio data

        Args:
            audiosamples: numpy array of audio samples at self.embedder.samplerate

        Returns:
            tensor results
        """
        embeddings = self.embedder.embed(audiosamples)
        results = self.model(embeddings)['dense']

        return results