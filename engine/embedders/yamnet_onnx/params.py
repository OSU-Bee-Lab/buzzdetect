"""Hyperparameters for the ONNX YAMNet front end.

A deliberate copy of the fields ``embedders/yamnet/params.py`` defines, so this
embedder can be bundled without dragging the TensorFlow embedder along with it.
These values were fixed when YAMNet was trained and have never moved; BUILD.py
re-checks the numpy front end against the TensorFlow one, so any drift between
the two copies shows up as a parity failure rather than as silently wrong
results.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    sample_rate: float = 16000.0
    stft_window_seconds: float = 0.025
    stft_hop_seconds: float = 0.010
    mel_bands: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0
    log_offset: float = 0.001
    patch_window_seconds: float = 0.96
    # Contiguous patches. Unlike the TensorFlow embedder, which can retune this
    # at load time, the ONNX graph is exported at a fixed [N, 96, 64] input, so
    # the hop is pinned to the window. See embedder.py.
    patch_hop_seconds: float = 0.96
