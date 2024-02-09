import multiprocessing
import os
import re
import sys

import librosa
import pandas as pd
import soundfile as sf


#  sniplist = [('./localData/a.wav', './localData/a_1.wav', 0 , 1), ('./localData/a.wav', './localData/a_2.wav', 2 , 3)]
def snip_audio(sniplist, cpus, conflict_out='skip', samplerate = 16000):
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
            samplerate_file = track.samplerate

            # snip loop
            #
            for path_raw, path_snip, start, end in sniplist_assigned:
                if os.path.exists(path_snip) and conflict_out == 'skip':
                    continue

                # print(f'snipper {worker_id}: snipping {path_snip}')
                start_frame = round(samplerate_file * start)
                frames_to_read = round(samplerate_file * (end - start))
                track.seek(start_frame)

                audio_data = track.read(frames_to_read)
                audio_data = librosa.resample(y=audio_data, orig_sr=samplerate_file, target_sr=samplerate)

                sf.write(path_snip, audio_data, samplerate)

    process_list = [multiprocessing.Process(target=worker_snipper, args=[c]) for c in range(cpus)]
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    print('snipping finished!')


def snip_training(path_annotations, dir_training, dir_raw, cpus, conflict_out='skip'):
    path_metadata = os.path.join(dir_training, "metadata", "metadata_raw.csv")
    dir_audio = os.path.join(dir_training, "audio")

    if os.path.exists(path_metadata):
        quit(f"metadata file already exists at {path_metadata}")

    os.makedirs(os.path.dirname(path_metadata), exist_ok=True)

    annotations = pd.read_csv(path_annotations)

    # kill leading slashes in ident if present; messes up os.path.join
    annotations['ident'] = [re.sub('^/', '', ident) for ident in annotations['ident']]
    annotations['path_raw'] = [os.path.join(dir_raw, ident + '.mp3') for ident in annotations['ident']]

    annotations['path_relative'] = [ident + '_s' + str(start.__floor__()) + '.wav' for ident, start in zip(annotations['ident'], annotations['start'])]
    annotations['duration'] = annotations['end'] - annotations['start']
    annotations['path_snip'] = [os.path.join(dir_audio, ident + "_s" + str(start.__floor__()) + ".wav") for ident, start in zip(annotations['ident'], annotations['start'])]

    annotations.to_csv(path_metadata, index=False)
    sniplist = list(zip(annotations['path_raw'], annotations['path_snip'], annotations['start'], annotations['end']))

    snip_audio(
        sniplist=sniplist,
        cpus=cpus,
        conflict_out='skip',
        samplerate=16000
    )


if __name__ == '__main__':
    snip_training(
        path_annotations='./localData/annotations/2024-02-05/annotations.csv',
        dir_training='./training',
        dir_raw='./localData/raw experiment audio',
        cpus=8,
        conflict_out='skip'
    )
