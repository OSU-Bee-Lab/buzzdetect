import multiprocessing
import os
import sys

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
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=resample_rate)
        sr = resample_rate

    return audio_data, sr


def frame_audio(audio_data, framelength, samplerate, framehop=0.5):
    framelength_samples = int(framelength*samplerate)
    audio_samples = len(audio_data)
    if audio_samples < framelength_samples:
        raise ValueError('sub-frame audio given')

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


def snip_audio(sniplist, cpus, conflict_out='skip'):
    """ takes sniplist as list of tuples (path_raw, path_snip, start, end) and cuts those snips out of larger raw
    audio files."""
    raws = list(set([t[0] for t in sniplist]))
    snips = list(set([t[1] for t in sniplist]))

    control_dict = {}
    for raw in raws:
        rawsnips = [t for t in sniplist if t[0] == raw]
        rawsnips = sorted(rawsnips, key=lambda x: x[2])  # sort for sequential seeking
        control_dict.update({raw: rawsnips})

    q_raw = multiprocessing.Queue()

    for raw in raws:
        q_raw.put(raw)

    for i in range(cpus):
        q_raw.put("terminate")

    dirs_out = list(set([os.path.dirname(snip) for snip in snips]))
    for d in dirs_out:
        os.makedirs(d, exist_ok=True)

    def worker_snipper(worker_id):
        print(f'snipper {worker_id}: starting')

        # Raw loop
        #
        while True:
            raw_assigned = q_raw.get()
            if raw_assigned == 'terminate':
                print(f"snipper {worker_id}: received terminate signal; exiting")
                sys.exit(0)

            print(f'snipper {worker_id}: starting on raw {raw_assigned}')
            sniplist_assigned = control_dict[raw_assigned]

            track = sf.SoundFile(raw_assigned)
            samplerate_native = track.samplerate

            # snip loop
            #
            for path_raw, path_snip, start, end in sniplist_assigned:
                if os.path.exists(path_snip) and conflict_out == 'skip':
                    continue

                # print(f'snipper {worker_id}: snipping {path_snip}')
                start_frame = round(samplerate_native * start)
                frames_to_read = round(samplerate_native * (end - start))
                track.seek(start_frame)

                audio_data = track.read(frames_to_read)

                sf.write(path_snip, audio_data, samplerate_native)

    process_list = [multiprocessing.Process(target=worker_snipper, args=[c]) for c in range(cpus)]
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    print('snipping finished!')
