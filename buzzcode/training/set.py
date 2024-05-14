import librosa
import multiprocessing
import pandas as pd
import soundfile as sf
from buzzcode.utils import save_pickle, setthreads
from buzzcode.audio import frame_audio
from buzzcode.analysis.analysis import melt_coverage
from buzzcode.utils import load_pickle
from datetime import datetime
import numpy as np
import json
import os

setthreads(1)

# TODO: generalize the whole multiprocessing thing...
def overlaps(range_frame, range_event, overlap_min):
    range_overlap = (range_frame[0] + overlap_min, range_frame[1] - overlap_min)

    # if event ends before frame starts or event starts after frame ends, False
    if (range_event[1] < range_overlap[0]) or (range_event[0] > range_overlap[1]):
        return False
    else:
        # otherwise, it must be overlapping, no matter how!
        return True


# TODO: make augmentation n-class capable, source/augment agnostic
def create_set_augment(annotationname, augmentationname=None, framehop=0.1, overlap_annotation=0.4, overlap_event=0.2, setname=None,
                   embeddername='yamnet', cpus=8):
    # per sample-of-interest:
        # add noise
        # change volume
        # overlap with other sounds
    lorem = []


def create_set_raw(annotationname, augmentationname=None, framehop=0.1, overlap_annotation=0.4, overlap_event=0.2, setname=None,
                   embeddername='yamnet', cpus=8):
    # TODO: make some check for overlap event needing to be X amount smaller than overlap annotation. Also figure out why that's the case.
    dir_annotations = './training/annotations'
    dir_raw = './localData/raw experiment audio'  # WITHOUT trailing slash, for creating path_raw

    if setname is None:
        setname = embeddername + '_' + annotationname

    dir_set = './training/sets/raw/' + setname
    os.makedirs(dir_set)

    annotations = pd.read_csv(os.path.join(dir_annotations, 'annotations_' + annotationname + '.csv')).dropna()
    classes = sorted(annotations['classification'].unique())



    annotations.to_csv(os.path.join(dir_set, 'annotations.csv'))

    idents = annotations['ident'].unique()
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
            print(f'{datetime.now()} constructor {worker_id}: starting ident {ident}')
            data_ident = []

            path_raw = dir_raw + ident + '.mp3'

            track = sf.SoundFile(path_raw)
            sr_native = track.samplerate
            audio_duration = librosa.get_duration(path=path_raw)

            sub = annotations[annotations['ident'] == ident].copy()

            coverage = melt_coverage(sub)

            for c, chunk in enumerate(coverage):
                print(f'{datetime.now()} constructor {worker_id}: starting chunk {(round(chunk[0], 1), round(chunk[1], 1))}')
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
                        f'{datetime.now()} constructor {worker_id}: unexpandable sub-frame audio; skipping')  # ugh, this could be avoided if I also framed backwards
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

                samples = [{'embeddings': emb, 'labels_raw': ev, 'ident': ident} for emb, ev in zip(embeddings, events)]
                data_ident.extend(samples)

            q_data.put(data_ident)
            ident = q_idents.get()

        print(f'{datetime.now()} constructor {worker_id}: terminate signal received; exiting')

    def worker_writer():
        alldata = []
        idents_remaining = len(idents)
        while idents_remaining > 0:
            print(f'writer: idents remaining: {idents_remaining}')
            data_ident = q_data.get()
            idents_remaining -= 1
            alldata.extend(data_ident)

        print(f'{datetime.now()} writer: all data received; writing')

        save_pickle(os.path.join(dir_set, 'data'), alldata)

        print(f'{datetime.now()} writer: saving config')
        config = {
            'setname': setname,
            'overlap_annotation': overlap_annotation,
            'overlap_event': overlap_event,
            'classes_raw': classes
        }

        config.update(q_config.get())

        with open(os.path.join(dir_set, 'config.txt'), 'x') as f:
            f.write(json.dumps(config))

        print({datetime.now()} 'writer: exiting')

    proc_writer = multiprocessing.Process(target=worker_writer)
    proc_writer.start()

    proc_constructors = []
    for a in range(cpus):
        proc_constructors.append(
            multiprocessing.Process(target=worker_constructor, name=f"constructor_proc{a}", args=([a])))
        proc_constructors[-1].start()
        pass

    proc_writer.join()


def events_to_targets(events, classes):
    y_blank = [0 for _ in classes]
    y_true = y_blank.copy()
    for event in events:
        y_true[classes.index(event)] = 1

    return np.array(y_true, dtype=np.int64)


def convert_labels(labels_raw, conversion_dict):
    try:
        labels_conv = [conversion_dict[l] for l in labels_raw if conversion_dict[l] is not np.nan]
    except KeyError:
        raise KeyError(f'label for sample not found in conversion; sample labels: {labels_raw}')

    return labels_conv


def translate_set(setname_raw, conversionname, setname_out=None):
    dir_set_raw = os.path.join('./training/sets/raw', setname_raw)
    conversion = pd.read_csv(os.path.join('./training/conversions', 'conversion_'+conversionname+'.csv'))
    conversion_dict = {f: t for f, t in zip(conversion['from'], conversion['to'])}

    classes = sorted(set(conversion['to'].dropna().to_list()))

    fold_df = pd.read_csv('./training/folds.csv')
    fold_dict = {i: f for i, f in zip(fold_df['ident'], fold_df['fold'])}

    set_raw = load_pickle(os.path.join(dir_set_raw, 'data'))

    def translate_sample(sample):
        labels_conv = convert_labels(sample['labels_raw'], conversion_dict)

        if labels_conv == []:
            return  # TODO: check that this appropriately drops elements

        fold = fold_dict[sample['ident']]
        targets = events_to_targets(labels_conv, classes)

        sample.update(
            {
                'labels_conv': labels_conv,  # I'm keeping the _conv tag on the key to prevent use of wrong set. Might reverse later.
                'fold': fold,
                'targets': targets
            }
        )
        del sample['labels_raw']  # maybe not necessary, but will save a bit of storage/memory

        return sample

    set_out = [translate_sample(s) for s in set_raw]
    set_out = [s for s in set_out if s is not None]

    if setname_out is None:
        setname_out = setname_raw + '_' + conversionname

    dir_set_out = os.path.join('./training/sets/translated/', setname_out)
    os.makedirs(dir_set_out)
    save_pickle(os.path.join(dir_set_out, 'data'), set_out)

    # Saving/copying associated data
    #
    annotations = pd.read_csv(os.path.join(dir_set_raw, 'annotations.csv'))
    annotations['classification_original'] = annotations['classification']
    annotations['classification'] = [conversion_dict[l] for l in annotations['classification_original']]
    annotations.to_csv(os.path.join(dir_set_out, 'annotations.csv'))

    conversion.to_csv(os.path.join(dir_set_out, 'conversion.csv'))

    with open(os.path.join(dir_set_raw, 'config.txt'), 'r') as f:
        config = json.load(f)
    config.update({'conversionname': conversionname})
    config.update({'classes': classes})
    with open(os.path.join(dir_set_out, 'config.txt'), 'x') as f:
        f.write(json.dumps(config))

