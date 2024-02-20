import librosa
import soundfile as sf
import numpy as np

def load_audio(path_audio, time_start=0, time_stop=None, resample_rate=None):
    track = sf.SoundFile(path_audio)

    can_seek = track.seekable() # True
    if not can_seek:
        raise ValueError("Input file not compatible with seeking")

    if time_stop is None:
        time_stop = librosa.get_duration(path=path_audio)

    sr = track.samplerate
    start_frame = round(sr * time_start)
    frames_to_read = round(sr * (time_stop - time_start))
    track.seek(start_frame)
    audio_data = track.read(frames_to_read)

    if resample_rate is not None:
        audio_resample = librosa.resample(y=audio_data, orig_sr=sr, target_sr=resample_rate)  # overwrite for memory purposes
        return audio_resample, resample_rate

    return audio_data, sr

path_audio = './localData/buzz_plane.wav'
audio_data, samplerate = load_audio(path_audio, time_stop=3)


def frame_audio(audio_data, framelength, samplerate, framehop=0.5):
    framelength_samples = int(framelength*samplerate)
    audio_samples = len(audio_data)
    if audio_samples < framelength_samples:
        quit("sub-frame audio given")

    frames = []
    frame_start = 0
    frame_end = framelength_samples
    step = int(framelength_samples*framehop)

    while frame_end <= audio_samples:
        frames.append(audio_data[frame_start:frame_end])
        frame_start += step
        frame_end += step

    return frames


def extract_frequencies(audio_data, sr=44100, n_freq=3, freq_range=(300, 600)):
    # n_fft = 4096
    n_fft = 8192
    freqs = np.array((np.arange(0, 1 + n_fft / 2) * sr) / n_fft)

    index_min = np.argmin(abs(freqs - freq_range[0]))
    index_max = np.argmin(abs(freqs - freq_range[1]))

    spectra = abs(librosa.stft(audio_data, n_fft=n_fft))

    spectrum = np.mean(spectra[index_min:index_max], axis=1)  # also consider mode or median

    spectrum_ispeak = np.array([spectrum[i-1] < spectrum[i] > spectrum[i+1] for i in range(1, len(spectrum) - 1)])

    peaks_indices = np.array([e[0] + 1 for e in enumerate(spectrum_ispeak) if e[1]])
    peak_amplitudes = spectrum[peaks_indices]

    # I need to check that index + index_min properly translates the indices back to the freqs
    try:
        # it's technically possible to get an error here if there are fewer than n_freq peaks in the spectrum
        max_indices = peaks_indices[np.argpartition(peak_amplitudes, -n_freq)[-n_freq:]]
        max_freqs = sorted(freqs[max_indices + index_min])
    except (IndexError, ValueError):
        max_indices = peaks_indices
        max_freqs = sorted(freqs[max_indices + index_min])
        max_freqs = np.append(max_freqs, [0 for _ in range(n_freq - len(max_freqs))])

    return max_freqs

