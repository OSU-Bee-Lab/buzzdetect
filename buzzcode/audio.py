import multiprocessing
import os
import sys
import warnings
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

    if audio_data.shape[1] > 1:  # if multi-channel, convert to mono
        audio_data = np.mean(audio_data, axis=-1)

    if resample_rate is not None:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=resample_rate)
        sr = resample_rate

    return audio_data, sr


def frame_audio(audio_data, framelength, samplerate, framehop_s):
    framelength_samples = int(framelength * samplerate)
    audio_samples = len(audio_data)
    if audio_samples < framelength_samples:
        raise ValueError('sub-frame audio given')

    step = int(framehop_s * samplerate)

    frames = []
    # yields consecutive audio frames, stopping before exceeding audio length
    for frame_start in range(0, audio_samples - framelength_samples + 1, step):
        frames.append(audio_data[frame_start:frame_start + framelength_samples])

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


def stream_to_queue(path_audio, chunklist, q_assignments, resample_rate, smallread_tolerance=0.98):
    def chunk_to_assignment(chunk, track, samplerate_native):
        sample_from = int(chunk[0] * samplerate_native)
        sample_to = int(chunk[1] * samplerate_native)
        read_size = sample_to - sample_from

        track.seek(sample_from)
        samples = track.read(read_size)
        if track.channels > 1:
            samples = np.mean(samples, axis=1)

        # we've found that many of our files give an incorrect .frames count, or else headers are broken
        # this results in a silent failure where no samples are returned
        n_samples = len(samples)
        if n_samples == 0:
            warnings.warn( f"no data read for chunk {chunk} in file {path_audio}")

        elif (n_samples/read_size) < smallread_tolerance:  # there's always a tiny smallread at end of file
            perc = int((n_samples / read_size) * 100)

            warnings.warn(
                f"unexpectedly small read for chunk {chunk} for file {path_audio}. "
                f"Received {perc}% of samples requested ({read_size}/{n_samples})")

        samples = librosa.resample(y=samples, orig_sr=samplerate_native, target_sr=resample_rate)

        assignment = {
            'path_audio': path_audio,
            'chunk': chunk,
            'samples': samples
        }

        q_assignments.put(assignment)

    track = sf.SoundFile(path_audio)
    samplerate_native = track.samplerate

    for chunk in chunklist:  # TODO: check for bad audio because samples returned is too small?
        chunk_to_assignment(chunk, track, samplerate_native)

    track.close()
