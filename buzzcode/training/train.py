from buzzcode.utils import save_pickle, setthreads
setthreads(1)
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
import json
from buzzcode.utils import save_pickle
from buzzcode.embeddings import get_embedder


# could maybe be generalized into frame_audio? also I'm sure this could get way faster, but kludge for now!
def count_frames(duration, framelength, framehop=0.5):
    frames = []
    frame_start = 0
    frame_end = framelength
    step = framelength*framehop

    while frame_end <= duration:
        frames.append((frame_start, frame_end))
        frame_start += step
        frame_end += step

    return len(frames)

def weight_inverseproportional(metadata, framelength):
    weightdf = pd.DataFrame()
    weightdf['classification'] = metadata['classification'].unique()

    def count_class_frames(classification_in):
        #  don't include test information in weighting
        sub = metadata[(metadata['classification'] == classification_in) & (metadata['fold'] != 5)]
        framecounts = [count_frames(d, framelength, 0.5) for d in sub['duration']]

        return sum(framecounts)

    weightdf['frames'] = [count_class_frames(c) for c in weightdf['classification']]

    frames_total = sum(weightdf['frames'])

    weightdf['weight'] = [np.log(frames_total / f) for f in weightdf['frames']]

    weightdf = weightdf.sort_values(by='classification')

    return weightdf


def weight_fractional(metadata, framelength):
    weightdf = pd.DataFrame()
    weightdf['classification'] = metadata['classification'].unique()

    def count_class_frames(classification_in):
        #  don't include test information in weighting
        sub = metadata[(metadata['classification'] == classification_in) & (metadata['fold'] != 5)]
        framecounts = [count_frames(d, framelength, 0.5) for d in sub['duration']]

        return sum(framecounts)

    weightdf['frames'] = [count_class_frames(c) for c in weightdf['classification']]

    frames_max = max(weightdf['frames'])

    weightdf['weight'] = [(frames_max / f) for f in weightdf['frames']]

    weightdf = weightdf.sort_values(by='classification')

    return weightdf


# modelname = 'test'; metadata_name="metadata_strict"; weights_name=None; epochs_in=3
def generate_model(modelname, embeddername='yamnet', metadata_name="metadata_strict", epochs_in=100):
    input_size = 1028  # not sure how to handle this; input shape will be fixed for any one model but is still in flux

    dir_model = os.path.join("./models/", modelname)
    if os.path.exists(dir_model) and modelname != 'test':
        raise FileExistsError(
            'a model folder with this name already exists; delete or rename the existing model folder and re-run')
    os.makedirs(dir_model, exist_ok=True)

    dir_training = './training'
    dir_metadata = os.path.join(dir_training, 'metadata')
    dir_audio = os.path.join(dir_training, 'audio')
    dir_cache = os.path.join(dir_training, 'input_cache_freq')

    embedder, config = get_embedder(embeddername)
    framelength = config['framelength']

    config.update({'input_size': input_size})

    # METADATA ----
    #

    # prep
    #
    if not bool(re.search('\\.csv$', metadata_name)):
        metadata_name = metadata_name + '.csv'
    metadata = pd.read_csv(os.path.join(dir_metadata, metadata_name))

    # trim leading slash, if present, from idents for os.path.join
    metadata['path_relative'] = [re.sub('^/', '', i) for i in metadata['path_relative']]

    if 'path_audio' not in metadata.columns:
        metadata['path_audio'] = [os.path.join(dir_audio, p) for p in metadata['path_relative']]

    if 'fold' not in metadata.columns:
        metadata['fold'] = np.random.randint(low=1, high=6, size=len(metadata))

    # drop inputs that haven't been cached
    # TODO: just generate the cache files!
    metadata['path_inputs'] = [re.sub(dir_audio, dir_cache, path) for path in metadata['path_audio']]
    metadata['path_inputs'] = [os.path.splitext(emb)[0] + '.npy' for emb in metadata['path_inputs']]

    metadata_noinputs = metadata[[not os.path.exists(p) for p in metadata['path_inputs']]]



    metadata = metadata[[os.path.exists(p) for p in metadata['path_inputs']]]

    # WEIGHTING ----
    #
    weightdf = weight_inverseproportional(metadata, framelength)
    weightdf.to_csv(os.path.join(dir_model, "weights.csv"), index=False)

    dict_weight = {index: weight for index, weight in enumerate(weightdf['weight'])}
    dict_names = {name: index for index, name in enumerate(weightdf['classification'])}

    config.update({'classes': weightdf['classification'].tolist()})

    # hmm this is throwing an error when it wasn't before; because subsetting for existing files?
    metadata['target'] = [dict_names[c] for c in metadata['classification']]  # add model internal target number

    # write metadata
    metadata.to_csv(os.path.join(dir_model, "metadata.csv"), index=False)

    # dataset creation
    #
    dataset_full = tf.data.Dataset.from_tensor_slices(
        (metadata['path_inputs'], metadata['target'], metadata['fold']))

    def load_embed(path_inputs, target, fold):
        embed_array = tf.numpy_function(
            func=lambda y: np.array(np.load(y.decode("utf-8"), allow_pickle=True)),
            inp=[path_inputs],
            Tout=tf.float64,
        )

        # convert numpy array to tensor
        embeddings = tf.convert_to_tensor(embeddings)
        n_embeddings = tf.shape(embeddings)[0]

        return embeddings, tf.repeat(target, n_embeddings), tf.repeat(fold, n_embeddings)

    dataset_full = dataset_full.map(lambda paths, labels, folds: load_embed(paths, labels, folds)).unbatch()

    def set_shape(embeddings, target, fold):
        embeddings.set_shape(input_size)

        return embeddings, target, fold

    dataset_full = dataset_full.map(lambda embeddings, labels, folds: set_shape(embeddings, labels, folds))

    # Split the data
    #
    dataset_cache = dataset_full.cache()
    dataset_train = dataset_cache.filter(lambda embedding, label, fold: fold < 4)
    dataset_val = dataset_cache.filter(lambda embedding, label, fold: fold == 4)

    # remove the folds column now that it's not needed anymore
    remove_fold_column = lambda embedding, label, fold: (embedding, label)
    dataset_train = dataset_train.map(remove_fold_column)
    dataset_val = dataset_val.map(remove_fold_column)

    dataset_train = dataset_train.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    dataset_val = dataset_val.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    # Model creation
    #
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(axis=-1, name='normalization'),
        tf.keras.layers.Input(shape=input_size, dtype=tf.float32, name='input'),
        tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(weightdf))
    ], name=modelname)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam",
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=3,
                                                restore_best_weights=True)

    history = model.fit(dataset_train,
                        epochs=epochs_in,  # would rather solely rely on early stopping
                        validation_data=dataset_val,
                        callbacks=callback,
                        class_weight=dict_weight)

    save_pickle(os.path.join(dir_model, 'history'), history)

    model.save(os.path.join(dir_model), include_optimizer=True)
    with open(os.path.join(dir_model, 'config.txt'), 'x') as f:
        f.write(json.dumps(config))


if __name__ == "__main__":
    modelprompt = input("Input model name: ")
    generate_model(modelprompt)
