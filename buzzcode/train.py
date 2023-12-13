# Imports
#
import os
import pandas as pd
import tensorflow as tf
import re
import numpy as np
import librosa
from buzzcode.tools import search_dir
from buzzcode.tools_tf import get_yamnet

# Loading YAMNet from TensorFlow Hub
#
yamnet_model = get_yamnet()


# I would rather use load_audio, but it isn't playing nice with the sliced tensorflow dataset;
# I get: TypeError: expected str, bytes or os.PathLike object, not Tensor
# probs because of extension line?
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor """  # I removed resampling as this should be done in preprocessing
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

# dir_training="./audio_training"
def get_snipDuration(dir_training, invalidate = False):
    paths_audio = search_dir(dir_training, ".wav")
    cachepath = os.path.join(dir_training, "durations.csv")

    df = pd.DataFrame()
    df['path'] = paths_audio

    if (not os.path.exists(cachepath)) or invalidate:
        df['duration'] = [librosa.get_duration(path=file) for file in df['path']]
        df.to_csv(cachepath)
        return df

    df_cache = pd.read_csv(cachepath)

    # drop cached values
    df = df[-df['path'].isin(df_cache['path'])]
    df['duration'] = [librosa.get_duration(path=file) for file in df['path']]

    df_out = pd.concat([df, df_cache[df_cache['path'].isin(paths_audio)]])

    return df_out


    # modelname="test"; epochs_in=80; dir_training="./audio_training"; drop_threshold = 0; path_weights=None


def generate_model(modelname, epochs_in, dir_training="./audio_training", drop_threshold=0, path_weights=None):
    dir_model = os.path.join("models/", modelname)

    if os.path.exists(dir_model):
        raise FileExistsError('a model folder with this name already exists; delete or rename the folder and re-run')

    os.makedirs(dir_model)

    # Acquiring and filtering training data
    #
    metadata = get_snipDuration(dir_training)
    metadata['classification'] = [re.search(string=file, pattern="_s\\d+_(.*)\\.wav").group(1) for file in metadata['path']]

    if drop_threshold > 0:
        classes = metadata.classification.unique()

        classes_keep = []
        for c in classes:
            count = len(metadata[metadata['classification'] == c])
            if count > drop_threshold:
                classes_keep.append(c)

        metadata = metadata[metadata['classification'].isin(classes_keep)]

    metadata['fold'] = np.random.randint(low=1, high=6, size=len(metadata))

    classes = metadata['classification'].unique()

    if path_weights is None:
        weightdf = metadata[['classification', 'duration']].groupby('classification').sum()
        weightdf['weight_raw'] = (sum(weightdf['duration'])/weightdf['duration'])
        weightdf['weight'] = weightdf['weight_raw']/sum(weightdf['weight_raw'])
        classes = list(weightdf.index)

    else:
        weightdf = pd.read_csv(path_weights)
        weights_new = pd.DataFrame()

        weights_new['classification'] = [c for c in classes if c not in weightdf['classification'].values]
        weights_new['weight'] = 1

        weightdf = pd.concat([weightdf, weights_new])
        classes = weightdf['classification']

    weightdf.to_csv(os.path.join(dir_model, "weights.csv"))

    weights = list(weightdf['weight'])

    map_class_to_weight = {index: weight for index, weight in enumerate(weights)}
    map_class_to_id = {name: index for index, name in enumerate(classes)}

    metadata['target_model'] = [map_class_to_id[c] for c in metadata['classification']]
    metadata.to_csv(os.path.join(dir_model, "metadata.csv"))

    # Load the audio files and retrieve embeddings
    #

    filenames = metadata['path']
    targets = metadata['target_model']
    folds = metadata['fold']

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

    def load_wav_for_map(filename, label, fold):
        return load_wav_16k_mono(filename), label, fold

    main_ds = main_ds.map(load_wav_for_map)

    # applies the embedding extraction model to a wav data
    def extract_embedding(wav_data, label, fold):
        ''' run YAMNet to extract embedding from the wav data '''
        scores, embeddings, spectrogram = yamnet_model(wav_data)
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
    test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

    # remove the folds column now that it's not needed anymore
    remove_fold_column = lambda embedding, label, fold: (embedding, label)

    train_ds = train_ds.map(remove_fold_column)
    val_ds = val_ds.map(remove_fold_column)
    test_ds = test_ds.map(remove_fold_column)

    train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

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
                        epochs=epochs_in,
                        validation_data=val_ds,
                        callbacks=callback,
                        class_weight=map_class_to_weight)

    model.save(os.path.join("models", modelname), include_optimizer=True)


if __name__ == "__main__":
    modelname = input("Input model name; 'test' for test run: ")
    if modelname == "test":
        if os.path.exists("./models/test"):
            import shutil

            shutil.rmtree("./models/test")
        generate_model("test", 1)

    generate_model(modelname, 50, drop_threshold=15, path_weights = "./audio_training/weight_test_rev3.csv")
