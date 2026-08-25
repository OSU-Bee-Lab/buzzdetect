"""NumPy port of the YAMNet log-mel front end.

The Keras YAMNet in ``embedders/yamnet`` computes its features inside the graph
(``WaveformFeatures`` -> ``features.py``), which means the framing, STFT and mel
projection all need TensorFlow at inference time. The ONNX export deliberately
starts *after* that front end, at the fixed [N, 96, 64] patch input, so this
module reproduces the same steps with nothing but NumPy.

Every constant and step here mirrors ``embedders/yamnet/features.py``; see
``BUILD.py`` for the parity check that holds the two implementations together.
"""

import numpy as np

# tf.signal.linear_to_mel_weight_matrix's constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def _hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                sample_rate, lower_edge_hertz,
                                upper_edge_hertz):
    """NumPy equivalent of tf.signal.linear_to_mel_weight_matrix."""
    # Bin 0 is the DC term, which is dropped (TF zeroes it out the same way).
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins)[bands_to_zero:]
    spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

    # num_mel_bins + 2 edges, read as overlapping (lower, center, upper)
    # triples -- the same thing tf.signal.frame(..., 3, 1) produces.
    band_edges_mel = np.linspace(_hertz_to_mel(lower_edge_hertz),
                                 _hertz_to_mel(upper_edge_hertz),
                                 num_mel_bins + 2)
    lower_edge_mel = band_edges_mel[np.newaxis, 0:-2]
    center_mel = band_edges_mel[np.newaxis, 1:-1]
    upper_edge_mel = band_edges_mel[np.newaxis, 2:]

    lower_slopes = ((spectrogram_bins_mel - lower_edge_mel) /
                    (center_mel - lower_edge_mel))
    upper_slopes = ((upper_edge_mel - spectrogram_bins_mel) /
                    (upper_edge_mel - center_mel))
    mel_weights = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-insert the zeroed DC row so the matrix lines up with the full
    # magnitude spectrogram.
    return np.pad(mel_weights, [[bands_to_zero, 0], [0, 0]]).astype(np.float32)


def pad_waveform(waveform, params):
    """NumPy port of features.pad_waveform.

    Pads with silence so the waveform yields a whole number of patches.
    """
    min_waveform_seconds = (params.patch_window_seconds +
                            params.stft_window_seconds -
                            params.stft_hop_seconds)
    min_num_samples = int(min_waveform_seconds * params.sample_rate)
    num_samples = waveform.shape[0]
    num_padding_samples = max(0, min_num_samples - num_samples)

    num_samples = max(num_samples, min_num_samples)
    num_samples_after_first_patch = num_samples - min_num_samples
    hop_samples = int(params.patch_hop_seconds * params.sample_rate)
    num_hops_after_first_patch = int(
        np.ceil(num_samples_after_first_patch / hop_samples))
    num_padding_samples += (hop_samples * num_hops_after_first_patch -
                            num_samples_after_first_patch)

    return np.pad(waveform, (0, num_padding_samples), mode='constant')


def _stft_magnitude(waveform, frame_length, frame_step, fft_length):
    """Magnitude of tf.signal.stft: periodic Hann, frames right-padded to
    fft_length, only complete frames emitted."""
    num_frames = 1 + (waveform.shape[0] - frame_length) // frame_step
    if num_frames < 1:
        return np.zeros((0, fft_length // 2 + 1), dtype=np.float32)

    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(num_frames, frame_length),
        strides=(waveform.strides[0] * frame_step, waveform.strides[0]),
    )
    # tf.signal.hann_window(..., periodic=True)
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(frame_length) / frame_length)
    windowed = frames * window.astype(np.float32)
    return np.abs(np.fft.rfft(windowed, n=fft_length)).astype(np.float32)


def waveform_to_patches(waveform, params):
    """Turn a 1-D waveform into [num_patches, 96, 64] log-mel patches.

    Equivalent to pad_waveform + waveform_to_log_mel_spectrogram_patches from
    embedders/yamnet/features.py, returning only the patches.
    """
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    waveform = pad_waveform(waveform, params)

    window_length_samples = int(round(params.sample_rate *
                                      params.stft_window_seconds))
    hop_length_samples = int(round(params.sample_rate *
                                   params.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1

    magnitude_spectrogram = _stft_magnitude(
        waveform, window_length_samples, hop_length_samples, fft_length)

    mel_matrix = linear_to_mel_weight_matrix(
        num_mel_bins=params.mel_bands,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.sample_rate,
        lower_edge_hertz=params.mel_min_hz,
        upper_edge_hertz=params.mel_max_hz)
    mel_spectrogram = magnitude_spectrogram @ mel_matrix
    log_mel_spectrogram = np.log(mel_spectrogram + params.log_offset)

    spectrogram_sample_rate = params.sample_rate / hop_length_samples
    patch_window_length = int(round(spectrogram_sample_rate *
                                    params.patch_window_seconds))
    patch_hop_length = int(round(spectrogram_sample_rate *
                                 params.patch_hop_seconds))

    num_patches = 1 + (log_mel_spectrogram.shape[0] -
                       patch_window_length) // patch_hop_length
    if num_patches < 1:
        return np.zeros((0, patch_window_length, params.mel_bands),
                        dtype=np.float32)

    # Only complete patches are emitted, matching tf.signal.frame. At
    # framehop 1 (hop == window) this is a plain non-overlapping view.
    return np.lib.stride_tricks.as_strided(
        log_mel_spectrogram,
        shape=(num_patches, patch_window_length, params.mel_bands),
        strides=(log_mel_spectrogram.strides[0] * patch_hop_length,
                 log_mel_spectrogram.strides[0],
                 log_mel_spectrogram.strides[1]),
    ).astype(np.float32)
