import os

import numpy as np

from src.inference.embedding import BaseEmbedder
from src.inference.onnx import make_session
from embedders.yamnet_onnx import features
from embedders.yamnet_onnx.params import Params


class EmbedderYamnetOnnx(BaseEmbedder):
    """YAMNet without TensorFlow.

    The TensorFlow embedder computes its log-mel front end inside the Keras
    graph. Here that front end runs in NumPy (features.py) and only the
    MobileNet trunk -- a fixed [N, 96, 64] patches -> [N, 1024] graph -- runs in
    onnxruntime. Numerically it matches the TensorFlow embedder to ~1e-5; see
    BUILD.py, which exports the graph and asserts that agreement.
    """

    embeddername = "yamnet_onnx"
    framelength_s = 0.96  # seconds
    digits_time = 2
    samplerate = 16000  # Hz
    n_embeddings = 1024
    dtype_in = 'float32'

    def __init__(self, framehop_prop):
        # The ONNX graph is exported with the patch hop welded to the patch
        # window, so an overlapping framehop can't be honoured here. Refuse it
        # outright rather than silently analysing on contiguous frames.
        if framehop_prop != 1:
            raise ValueError(
                f'{self.embeddername} only supports framehop_prop=1, got '
                f'{framehop_prop}. Use the tensorflow "yamnet" embedder for '
                f'overlapping frames.')
        super().__init__(framehop_prop)
        self.params = Params()

    def initialize(self):
        curdir = os.path.dirname(os.path.realpath(__file__))
        path_onnx = os.path.join(curdir, 'yamnet.onnx')
        self.model = make_session(path_onnx, self.processor)
        self.name_in = self.model.get_inputs()[0].name

    def embed(self, audio):
        """Generate embeddings for a 1-D waveform at self.samplerate.

        Args:
            audio: numpy array of audio samples at self.samplerate

        Returns:
            numpy array of shape [num_frames, 1024]
        """
        patches = features.waveform_to_patches(np.asarray(audio), self.params)
        return self.model.run(None, {self.name_in: patches})[0]
