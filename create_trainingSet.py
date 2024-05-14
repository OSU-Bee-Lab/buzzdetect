import os
import re
import sys
import json
import librosa
import multiprocessing
import numpy as np
import pandas as pd
import soundfile as sf
from buzzcode.utils import save_pickle, setthreads
from buzzcode.audio import frame_audio
from buzzcode.analysis.analysis import melt_coverage
setthreads(1)


def overlaps(range_frame, range_event, overlap_min):
    range_overlap = (range_frame[0] + overlap_min, range_frame[1] - overlap_min)

    # if event ends before frame starts or event starts after frame ends, False
    if (range_event[1] < range_overlap[0]) or (range_event[0] > range_overlap[1]):
        return False
    else:
        # otherwise, it must be overlapping, no matter how!
        return True


# dir_annotations = './localData/annotations/2024-03-15'; metadataname = 'metadata_intermediate_new.csv'; framehop=0.2; overlap_annotation=0.4; overlap_event=0.1; embeddername='yamnet'; cpus=7; setname=None
def cache_training(dir_annotations, metadataname, framehop=0.2, overlap_annotation=0.4, overlap_event=0.1, setname = None,
                   embeddername='yamnet', cpus=4):
    metadataname = os.path.splitext(metadataname)[0]  # drop file extension, if given

    if setname is None:
        setname = embeddername + '_' + re.search('metadata_(.*)', metadataname).group(1)

    dir_set = './training/' + setname
    os.makedirs(dir_set)

    metadata = pd.read_csv(os.path.join(dir_annotations, metadataname + '.csv'))
    classes = sorted(metadata['classification'].unique())

    metadata.to_csv(os.path.join(dir_set, 'metadata.csv'))
    dir_raw = './localData/raw experiment audio'  # WITHOUT trailing slash, for creating path_raw

    idents = metadata['ident'].unique()
    q_idents = multiprocessing.Queue()

    for i in idents:
        q_idents.put(i)

    for _ in range(cpus):
        q_idents.put('terminate')

    q_data = multiprocessing.Queue()
    q_config = multiprocessing.Queue()  # TODO: use more appropriate method than queue!

    def worker_constructor(worker_id):
        # TODO: figure out why I can't import get_embedder in parent context!
        from buzzcode.embeddings import get_embedder
        embedder, config = get_embedder(embeddername)
        q_config.put(config)

        framelength = config['framelength']
        annotation_extension = ((1 - overlap_annotation) * framelength)

        ident = q_idents.get()
        while ident != 'terminate':
            print(f'constructor {worker_id}: starting ident {ident}')
            data_ident = []

            path_raw = dir_raw + ident + '.mp3'

            track = sf.SoundFile(path_raw)
            sr_native = track.samplerate
            audio_duration = librosa.get_duration(path=path_raw)

            sub = metadata[metadata['ident'] == ident].copy()
            fold = sub['fold'].unique()
            if len(fold) > 1:
                print(f'constructor {worker_id}: FATAL ERROR, multiple folds for {ident}')
                sys.exit(0)

            coverage = melt_coverage(sub)

            for c, chunk in enumerate(coverage):
                print(f'constructor {worker_id}: starting chunk {(round(chunk[0], 1), round(chunk[1], 1))}')
                start = max(
                    chunk[0] - annotation_extension,
                    0  # if start < 0, round up to 0
                )

                end = min(
                    chunk[1] + (annotation_extension * 1.1),  # rounding makes the extension slightly low sometimes
                    audio_duration  # if end > file length, round down to file length
                )

                if (end - start) < framelength:
                    print(
                        f'constructor {worker_id}: unexpandable sub-frame audio; skipping')  # ugh, this could be avoided if I also framed backwards
                    continue

                start_sample = round(sr_native * start)
                samples_to_read = round(sr_native * (end - start))
                track.seek(start_sample)

                audio_data = track.read(samples_to_read)
                audio_data = librosa.resample(y=audio_data, orig_sr=sr_native, target_sr=config['samplerate'])

                frames = frame_audio(
                    audio_data=audio_data,
                    framelength=framelength,
                    samplerate=config['samplerate'],
                    framehop=framehop
                )

                embeddings = embedder(frames)

                events = []
                for i, frame in enumerate(frames):
                    s = start + (i * framehop * framelength)
                    e = s + framelength

                    sub['overlaps'] = [overlaps((s, e), (e_start, e_end), framelength * overlap_event) for
                                       e_start, e_end in zip(sub['start'], sub['end'])]
                    e = sub[sub['overlaps']]['classification'].unique().tolist()
                    if e == []:
                        print(f'!!! NULL EVENT IN {ident} {chunk}, FRAME {i} !!!')

                    events.append(e)

                data_ident.extend(list(zip(embeddings, events, [fold]*len(frames))))

            q_data.put(data_ident)

            ident = q_idents.get()

        print(f'constructor {worker_id}: terminate signal received; exiting')

    def worker_writer():
        alldata = []
        idents_remaining = len(idents)
        while idents_remaining > 0:
            print(f'writer: idents remaining: {idents_remaining}')
            data_ident = q_data.get()
            idents_remaining -= 1
            alldata.extend(data_ident)

        print('writer: all data received; writing')
        inputs, events, folds = zip(*alldata)  # outputs as tuples

        y_blank = [0 for _ in classes]

        def events_to_targets(events_in):
            y_true = y_blank.copy()
            for event in events_in:
                y_true[classes.index(event)] = 1

            return np.array(y_true, dtype=np.int64)

        targets = [events_to_targets(e) for e in events]

        data_full = (list(inputs), targets, list(events), list(folds))

        save_pickle(os.path.join(dir_set, 'data_cache'), data_full)

        print('writer: saving config')
        config = {
            'setname': setname,
            'overlap_annotation': overlap_annotation,
            'overlap_event': overlap_event,
            'classes': classes
        }

        config.update(q_config.get())

        with open(os.path.join(dir_set, 'config.txt'), 'x') as f:
            f.write(json.dumps(config))

        print('writer: exiting')

    proc_writer = multiprocessing.Process(target=worker_writer)
    proc_writer.start()

    proc_constructors = []
    for a in range(cpus):
        proc_constructors.append(
            multiprocessing.Process(target=worker_constructor, name=f"constructor_proc{a}", args=([a])))
        proc_constructors[-1].start()
        pass

    proc_writer.join()


if __name__ == '__main__':
    # full re-set
    cpus = 2
    cache_training(
        dir_annotations='./localData/annotations/2024-03-15',
        metadataname='metadata_intermediate_new',
        cpus=cpus
    )

    cache_training(
        dir_annotations='./localData/annotations/2024-03-15',
        metadataname='metadata_intermediate',
        cpus=cpus
    )

    cache_training(
        dir_annotations='./localData/annotations/2024-03-15',
        metadataname='metadata_strict',
        cpus=cpus
    )
