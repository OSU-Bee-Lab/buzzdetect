"""Export embedders/yamnet/yamnet.keras to yamnet.onnx, and check parity.

Run from the buzzdetect root, in an environment that has TensorFlow:

    .venv/bin/python3 embedders/yamnet_onnx/BUILD.py

The Keras model is waveform -> 1024 embeddings, with the log-mel front end
(WaveformFeatures) as its first layer. The ONNX export drops that layer: the
front end is reimplemented in NumPy in features.py, and only the MobileNet
trunk (a fixed [N, 96, 64] -> [N, 1024] graph) is exported. Doing it this way
avoids exporting a variable-length streaming graph, and makes the patch hop a
plain slicing stride instead of a mutable graph parameter.

The script fails loudly if the two paths disagree, so it doubles as the
regression test that keeps features.py honest.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import keras
import tensorflow as tf
import onnxruntime as ort

import embedders.yamnet.features as tf_features
from embedders.yamnet.params import Params as ParamsTF
from embedders.yamnet.yamnet import WaveformFeatures  # noqa: F401  (needed to deserialize yamnet.keras)
from embedders.yamnet_onnx import features as np_features
from embedders.yamnet_onnx.params import Params

DIR_EMBEDDER = os.path.dirname(os.path.realpath(__file__))
PATH_KERAS = os.path.join(os.path.dirname(DIR_EMBEDDER), 'yamnet', 'yamnet.keras')
PATH_ONNX = os.path.join(DIR_EMBEDDER, 'yamnet.onnx')

# Loose enough for float32 reassociation between two runtimes, tight enough to
# catch a genuinely wrong mel matrix, window or framing.
TOL_PATCHES = 1e-3
TOL_EMBEDDINGS = 1e-3


def build_core(model_full):
    """Rebuild the Keras model without its waveform front end.

    Layer 0 is the waveform Input and layer 1 is WaveformFeatures; everything
    from the Reshape onward is the MobileNet trunk, which is a plain linear
    stack. Reusing the layer objects carries their trained weights across.
    """
    patches = keras.Input(shape=(96, 64), name='patches', dtype='float32')
    x = patches
    for layer in model_full.layers[2:]:
        x = layer(x)
    core = keras.Model(patches, x, name='yamnet_core')
    # Keras refuses to export a model it has never seen called.
    core(np.zeros((1, 96, 64), dtype=np.float32))
    return core


def main():
    params = Params()
    tf_params = ParamsTF(patch_hop_seconds=params.patch_hop_seconds)

    print(f'loading {PATH_KERAS}')
    model_full = keras.models.load_model(PATH_KERAS, compile=False)
    # yamnet.keras was saved with the stock 0.48s patch hop; the ONNX path is
    # contiguous-only, so retune the front end before comparing the two.
    model_full.layers[1].params.patch_hop_seconds = params.patch_hop_seconds
    core = build_core(model_full)

    print(f'exporting {PATH_ONNX}')
    core.export(PATH_ONNX, format='onnx', verbose=False)
    session = ort.InferenceSession(PATH_ONNX, providers=['CPUExecutionProvider'])
    name_in = session.get_inputs()[0].name
    print(f'  {os.path.getsize(PATH_ONNX) / 1e6:.1f} MB, '
          f'input {session.get_inputs()[0].shape}, '
          f'output {session.get_outputs()[0].shape}')

    print('checking parity against the tensorflow embedder')
    rng = np.random.default_rng(0)
    worst_patches = 0.0
    worst_embeddings = 0.0
    # Lengths chosen to cover: several whole patches, exactly one patch, one
    # sample under the padding floor, a ragged tail, and a clip too short to
    # make a patch at all.
    for n_samples in (16000 * 30, 15360, 15599, 16000 * 7 + 137, 4000):
        waveform = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)

        padded = tf_features.pad_waveform(tf.convert_to_tensor(waveform), tf_params)
        _, patches_tf = tf_features.waveform_to_log_mel_spectrogram_patches(
            padded, tf_params)
        patches_tf = patches_tf.numpy()
        patches_np = np_features.waveform_to_patches(waveform, params)

        if patches_tf.shape != patches_np.shape:
            raise SystemExit(
                f'patch shape mismatch at n={n_samples}: '
                f'tensorflow {patches_tf.shape} vs numpy {patches_np.shape}')

        embeddings_tf = model_full(waveform).numpy()
        embeddings_onnx = session.run(None, {name_in: patches_np})[0]

        d_patches = float(np.abs(patches_tf - patches_np).max())
        d_embeddings = float(np.abs(embeddings_tf - embeddings_onnx).max())
        worst_patches = max(worst_patches, d_patches)
        worst_embeddings = max(worst_embeddings, d_embeddings)
        print(f'  n={n_samples:<8} patches={patches_np.shape}  '
              f'max|dpatch|={d_patches:.2e}  max|dembed|={d_embeddings:.2e}')

    if worst_patches > TOL_PATCHES or worst_embeddings > TOL_EMBEDDINGS:
        raise SystemExit(
            f'parity check FAILED: patches {worst_patches:.2e} '
            f'(tol {TOL_PATCHES}), embeddings {worst_embeddings:.2e} '
            f'(tol {TOL_EMBEDDINGS})')

    print(f'parity OK: patches {worst_patches:.2e}, '
          f'embeddings {worst_embeddings:.2e}')


if __name__ == '__main__':
    main()
