# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Core model definition of YAMNet."""

import csv
import keras

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

import embedders.yamnet.features as features_lib


def _batch_norm(name, params):
  def _bn_layer(layer_input):
    return layers.BatchNormalization(
      name=name,
      center=params.batchnorm_center,
      scale=params.batchnorm_scale,
      epsilon=params.batchnorm_epsilon)(layer_input)
  return _bn_layer


def _conv(name, kernel, stride, filters, params):
  def _conv_layer(layer_input):
      # CHANGE: changed "/" in layer names to "_" for keras 3
    output = layers.Conv2D(name='{}_conv'.format(name),
                           filters=filters,
                           kernel_size=kernel,
                           strides=stride,
                           padding=params.conv_padding,
                           use_bias=False,
                           activation=None)(layer_input)
    output = _batch_norm('{}_conv_bn'.format(name), params)(output)
    output = layers.ReLU(name='{}_relu'.format(name))(output)
    return output
  return _conv_layer


def _separable_conv(name, kernel, stride, filters, params):
  def _separable_conv_layer(layer_input):
  # CHANGE: changed "/" in layer names to "_" for keras 3
    output = layers.DepthwiseConv2D(name='{}_depthwise_conv'.format(name),
                                    kernel_size=kernel,
                                    strides=stride,
                                    depth_multiplier=1,
                                    padding=params.conv_padding,
                                    use_bias=False,
                                    activation=None)(layer_input)
    output = _batch_norm('{}_depthwise_conv_bn'.format(name), params)(output)
    output = layers.ReLU(name='{}_depthwise_conv_relu'.format(name))(output)
    output = layers.Conv2D(name='{}_pointwise_conv'.format(name),
                           filters=filters,
                           kernel_size=(1, 1),
                           strides=1,
                           padding=params.conv_padding,
                           use_bias=False,
                           activation=None)(output)
    output = _batch_norm('{}_pointwise_conv_bn'.format(name), params)(output)
    output = layers.ReLU(name='{}_pointwise_conv_relu'.format(name))(output)
    return output
  return _separable_conv_layer


_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]


def yamnet(features, params):
  """Define the core YAMNet mode in Keras."""
  net = layers.Reshape(
      (params.patch_frames, params.patch_bands, 1),
      input_shape=(params.patch_frames, params.patch_bands))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
  embeddings = layers.GlobalAveragePooling2D()(net)
  logits = layers.Dense(units=params.num_classes, use_bias=True)(embeddings)
  predictions = layers.Activation(activation=params.classifier_activation)(logits)
  return predictions, embeddings


# CHANGE: added this class to handle:
    # ValueError: A KerasTensor cannot be used as input to a TensorFlow function
    # when execuing the original code:   waveform_padded = features_lib.pad_waveform(waveform, params)
@keras.saving.register_keras_serializable()
class WaveformFeatures(layers.Layer):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params

    def get_config(self):
        base = super().get_config()

        def to_jsonable(obj):
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            if isinstance(obj, (list, tuple)):
                return [to_jsonable(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): to_jsonable(v) for k, v in obj.items()}
            if hasattr(obj, "__dict__"):
                # Convert simple objects (like Params) to dict
                return {str(k): to_jsonable(v) for k, v in vars(obj).items()}
            # Fallback: string representation
            return str(obj)

        params_dict = to_jsonable(self.params)
        return {**base, "params": params_dict}

    @classmethod
    def from_config(cls, config):
        params_cfg = config.get("params", {})
        try:
            # Try to reconstruct the canonical Params() object
            from params import Params as _Params
            p = _Params()
            for k, v in params_cfg.items():
                if hasattr(p, k):
                    setattr(p, k, v)
            return cls(p)
        except Exception:
            # Fallback: use a simple namespace with the same attributes
            import types
            p = types.SimpleNamespace(**params_cfg)
            return cls(p)

    def call(self, waveform):
        padded = features_lib.pad_waveform(waveform, self.params)
        log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
            padded, self.params)
        return log_mel_spectrogram, features


def yamnet_frames_model(params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of Params containing hyperparameters.

  Returns:
    A model accepting (num_samples,) waveform input and emitting:
    - predictions: (num_patches, num_classes) matrix of class scores per time frame
    - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
    - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
  """
  # CHANGE: overwrote first four lines of this function ot use new WaveformFeatures class
  waveform = layers.Input(shape=(), dtype=tf.float32, name='waveform')
  log_mel_spectrogram, features = WaveformFeatures(params)(waveform)
  predictions, embeddings = yamnet(features, params)
  frames_model = Model(
      name='yamnet_frames', inputs=waveform,
      outputs=[predictions, embeddings, log_mel_spectrogram])
  return frames_model
