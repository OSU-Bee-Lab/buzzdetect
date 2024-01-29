# Imports
#
import sys

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import librosa
from buzzcode.utils.analysis import get_yamnet, load_audio_tf
from buzzcode.test_model import analyze_testFold

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Loading YAMNet from TensorFlow Hub
#
yamnet = get_yamnet()


def generate_model(modelname, path_metadata="./training/metadata/metadata_raw.csv", path_weights=None, test_model=False, cpus=None, memory_allot=None, max_per_class=100):
    if test_model and (cpus is None or memory_allot is None):
        sys.exit("cpu count and memory allotment must be given when testing model")

    dir_model = os.path.join("./models/", modelname)

    if os.path.exists(dir_model):
        raise FileExistsError('a model folder with this name already exists; delete or rename the existing model folder and re-run')

    os.makedirs(dir_model)

    # Acquiring and filtering training data
    #
    metadata = pd.read_csv(path_metadata)

    if 'path' not in metadata.columns:
        metadata['path'] = [os.path.join('./training/audio', p) for p in metadata['path_relative']]

    if 'fold' not in metadata.columns:
        metadata['fold'] = np.random.randint(low=1, high=6, size=len(metadata))

    classes = metadata['classification'].unique()

    if path_weights is None:
        weightdf = pd.DataFrame()
        weightdf['classification'] = metadata['classification'].unique()
        weightdf['weight'] = 1
        classes = weightdf['classification'].to_list()

    else:
        weightdf = pd.read_csv(path_weights)
        weights_new = pd.DataFrame()

        # if a weight isn't assigned to a class, assign it 1
        weights_new['classification'] = [c for c in classes if c not in weightdf['classification'].values]
        weights_new['weight'] = 1

        weightdf = pd.concat([weightdf, weights_new])
        classes = weightdf['classification']

    weightdf.to_csv(os.path.join(dir_model, "weights.csv"))

    weights = list(weightdf['weight'])
    dict_weight = {index: weight for index, weight in enumerate(weights)}
    dict_names = {name: index for index, name in enumerate(classes)}

    metadata['target'] = [dict_names[c] for c in metadata['classification']]
    metadata.to_csv(os.path.join(dir_model, "metadata.csv"))

    # Load the audio files and retrieve embeddings
    #

    filenames = metadata['path']
    targets = metadata['target']
    folds = metadata['fold']

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

    def load_wav_for_map(filename, label, fold):
        return load_audio_tf(filename), label, fold

    main_ds = main_ds.map(load_wav_for_map)

    # returns YAMNet embeddings for given data
    def extract_embedding(wav_data, label, fold):
        scores, embeddings, spectrogram = yamnet(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (embeddings,
                tf.repeat(label, num_embeddings),
                tf.repeat(fold, num_embeddings))

    # extract embedding
    main_ds = main_ds.map(extract_embedding).unbatch()

    # Split the data
    #

    cached_ds = main_ds.cache()
    train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
    val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
    #
    # # remove the folds column now that it's not needed anymore
    remove_fold_column = lambda embedding, label, fold: (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    #
    # Create your model
    #

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                              name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(classes))
    ], name=modelname)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam",
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=3,
                                                restore_best_weights=True)

    history = model.fit(train_ds,
                        epochs=100,  # would rather solely rely on early stopping
                        validation_data=val_ds,
                        callbacks=callback,
                        class_weight=dict_weight)

    model.save(os.path.join("models", modelname), include_optimizer=True)

    if test_model:
        print("analyzing testfold data")
        analyze_testFold(modelname, cpus, memory_allot)

if __name__ == "__main__":
    # modelname = input("Input model name; 'test' for test run: ")
    modelname = "test"
    if modelname == "test":
        if os.path.exists("./models/test"):
            import shutil

            shutil.rmtree("./models/test")
        generate_model("test")

    generate_model(modelname, drop_threshold=400, path_weights="./weights.csv", test_model=True, cpus=8, memory_allot=8)


