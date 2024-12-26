import librosa
import multiprocessing
import pandas as pd
import soundfile as sf
import pickle
from buzzcode.training.set import overlapping_elements, expand_chunk
from buzzcode.audio import frame_audio
from buzzcode.utils import setthreads
from buzzcode.analysis import melt_coverage
from itertools import product, cycle
import numpy as np
import json
import os
import re

setthreads(1)

dir_annotations = './training/annotations'

dir_sets = './training/sets'
dir_sets_raw = os.path.join(dir_sets, 'raw')

dir_augmentation = './training/augmentation'

path_folds = './training/folds.csv'

dir_audio = './localData/raw experiment audio'  # WITHOUT trailing slash


def augment_combine(frames_source, frames_augment, limit='square'):
    """
        Combines source frames with augmentation frames while avoiding temporal correlations.
        Uses two different strategies depending on the size of frames_augment:
        1. When len(frames_augment) < limit: Makes every combination of source and augment frames
        2. Else: Makes random combinations of frames_source and frames_augment (where each combination is unique)

        Args:
            frames_source: List of source frames to be augmented
            frames_augment: List of frames to use for augmentation
            limit: Either 'square' or an integer limiting augmentations per source frame
        """

    if limit == 'square':
        limit = len(frames_source)
    n_frames_augment = len(frames_augment)

    # if we aren't over our limit on augment frames, we'll end up using them all. Just yield the product.
    if n_frames_augment <= limit:
        for source, augment in product(frames_source, frames_augment):
            yield (source + augment) / 2

    # if we are over the limit on augment frames, we need to make sure we don't end up with weird correlations,
    # e.g., one frame of source is augmented entirely with augment frames from the same event (one minute of rainstorm)
    shuffled_augment = frames_augment.copy()
    np.random.shuffle(shuffled_augment)
    augment_cycler = cycle(shuffled_augment)

    for frame_source in frames_source:
        for _ in range(limit):
            frame_augment = next(augment_cycler)
            yield (frame_source + frame_augment) / 2

def extract_samples(ident, annotations, config, classes_to_augment, framehop=None):
    print(f'starting ident {ident}')
    if framehop is None:
        framehop = config['framehop']

    path_raw = re.sub('/$', '', dir_audio) + ident + '.mp3'

    track = sf.SoundFile(path_raw)
    sr_native = track.samplerate
    audio_duration = librosa.get_duration(path=path_raw)

    sub = annotations[annotations['ident'] == ident].copy()

    def process_chunk(chunk):
        print(f'{ident}: starting chunk {(round(chunk[0], 1), round(chunk[1], 1))}')

        start_sample = round(sr_native * chunk[0])
        samples_to_read = round(sr_native * (chunk[1] - chunk[0]))
        track.seek(start_sample)

        audio_data = track.read(samples_to_read, dtype='float32')
        audio_data = librosa.resample(y=audio_data, orig_sr=sr_native, target_sr=config['samplerate'])

        frames = frame_audio(audio_data=audio_data, framelength=config['framelength'], samplerate=config['samplerate'],
                             framehop=framehop)

        range_table = list(zip(sub['start'], sub['end']))

        frametimes = chunk[0] + (np.arange(0, len(frames)) * (config['framelength'] * framehop))
        frametimes = [(s, s + config['framelength']) for s in frametimes]
        events_chunk = []

        for frame_range in frametimes:
            events_frame = overlapping_elements(
                range_index=frame_range,
                range_table=range_table,
                elements=sub['classification'],
                minimum_overlap=config['overlap_event'] * config['framelength']
            ).unique().tolist()

            events_chunk.append(events_frame)

        samples_chunk = [{'audio_data': f, 'label_raw': e} for f, e in zip(frames, events_chunk)]
        samples_chunk = [c for c in samples_chunk if len(c['label_raw']) == 1]
        for c in samples_chunk:
            c['label_raw'] = c['label_raw'][0]

        return samples_chunk

    chunks_raw = melt_coverage(sub)
    chunks_expand = [
        expand_chunk(chunk, config['framelength'], config['overlap_event'] * config['framelength'], audio_duration) for
        chunk in chunks_raw]

    samples_ident = []
    for chunk in chunks_expand:
        samples_ident.extend(process_chunk(chunk))

    classframes_ident = {k: [] for k in classes_to_augment}
    for sample in samples_ident:
        classframes_ident[sample['label_raw']].append(sample['audio_data'])

    return classframes_ident


def create_augment(name_set_raw, name_augmentation, cpus=8, framehop=None, limit='square'):
    # TODO:
    # check if augmentation data is already present! As-is, would just append more data on top
    # add add augmentation with white noise (should apply to negative cases, too)
    # change volume
    # change frequency response?
    dir_set_raw = os.path.join(dir_sets_raw, name_set_raw)
    with open(os.path.join(dir_set_raw, 'config.txt'), 'r') as f:
        config_set = json.load(f)

    name_augmentation = re.sub('^augmentation_', '', name_augmentation)
    name_augmentation = re.sub('\\.csv', '', name_augmentation)
    augmentation = pd.read_csv(os.path.join(dir_augmentation, 'augmentation_' + name_augmentation + '.csv'))
    classes_to_augment = set(augmentation['source'].tolist() + augmentation['augment'].tolist())

    with open(os.path.join(dir_set_raw, 'config_augment.txt'), 'x') as file:
        file.write(json.dumps({'framehop': framehop, 'limit': limit}))

    fold_df = pd.read_csv(path_folds)
    fold_df = fold_df[[fold < 4 for fold in fold_df['fold']]]  # drop test, train

    annotations = pd.read_csv(os.path.join(dir_set_raw, 'annotations.csv'))
    annotations = annotations[[c in classes_to_augment for c in annotations['classification']]]
    annotations = annotations[[i in fold_df['ident'].tolist() for i in annotations['ident']]]  # don't augment test or train

    idents = annotations['ident'].unique()

    with multiprocessing.Pool(processes=cpus) as pool:
        args = [(ident, annotations, config_set, classes_to_augment, framehop) for ident in idents]
        results = pool.starmap_async(extract_samples, args)
        pool.close()
        pool.join()
        classframe_list = results.get()

    classframes = {k: [] for k in classes_to_augment}
    for classframes_ident in classframe_list:
        for k in classframes_ident.keys():
            classframes[k].extend(classframes_ident[k])

    def construct_combined(class_source, class_augment, embedder):
        path_augment = os.path.join(dir_set_raw, 'augment_combine_' + class_source + '_' + class_augment)  # ugly name but *shruggie*

        frames_source = classframes[class_source]
        frames_augment = classframes[class_augment]
        frames_combined = augment_combine(frames_source, frames_augment, limit=limit)
        with open(path_augment, 'wb') as file:
            for f in frames_combined:
                embeddings = embedder(f)
                # TODO: this makes embeddings as tf.Tensor: shape=(1, 1024), dtype=float32;
                # TODO: but we need tf.Tensor: shape=(1024,), dtype=float32 (I think...)
                # could tf.squeeze
                sample = {'embeddings': embeddings, 'labels_raw': [class_source, class_augment]}
                pickle.dump(sample, file)

    q_augmentations_combined = multiprocessing.Queue()
    for c in zip(augmentation['source'], augmentation['augment']):
        q_augmentations_combined.put(c)
    for c in range(cpus):
        q_augmentations_combined.put('terminate')

    def worker_combiner(worker_id):
        # makes more sense to do this with starmap, but pickling + tensorflow is a nightmare
        print(f'combiner {worker_id} loading embedder')
        from buzzcode.embeddings import load_embedder
        embedder, _ = load_embedder(config_set['embeddername'])

        c = q_augmentations_combined.get()
        while c != 'terminate':
            print(f'combiner {worker_id} combining {c}')
            construct_combined(c[0], c[1], embedder)
            c = q_augmentations_combined.get()

        print(f'combiner {worker_id} received terminate signal; exiting')

    proc_combiners = []
    for c in range(cpus):
        proc_combiners.append(multiprocessing.Process(target=worker_combiner, args=[c]))
        proc_combiners[-1].start()

    for proc_combiner in proc_combiners:
        proc_combiner.join()


if __name__ == '__main__':
    create_augment('yamnet_droptraffic', 'buzz', cpus=7, framehop=0.25, limit=10)
