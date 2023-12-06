# Imports
#
import os
import pandas as pd
import tensorflow as tf
import re
import numpy as np
import librosa
from buzzcode.tools import load_audio, search_dir, get_yamnet

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


# modelName="test"; epochs_in=80; dir_training="./audio_training"; drop_threshold = 0
def generate_model(modelName, epochs_in, dir_training="./audio_training", drop_threshold=0):
    dir_model = os.path.join("models/", modelName)

    if os.path.exists(dir_model):
        raise FileExistsError('a model folder with this name already exists; delete or rename the folder and re-run')

    os.makedirs(dir_model)

    # Acquiring and filtering training data
    #
    audio_files = search_dir(dir_training, "wav")

    metadata = pd.DataFrame()
    metadata['path'] = audio_files
    metadata['classification'] = [re.search(string=file, pattern="_s\\d+_(.*)\\.wav").group(1) for file in audio_files]

    if drop_threshold > 0:
        classes = metadata.classification.unique()

        classes_keep = []
        for c in classes:
            count = len(metadata[metadata['classification'] == c])
            if count > drop_threshold:
                classes_keep.append(c)

        metadata = metadata[metadata['classification'].isin(classes_keep)]

    classes = metadata.classification.unique()

    with open(os.path.join(dir_model, "classes.txt"), "w") as file:
        for item in classes:
            file.write(item + "\n")

    metadata['fold'] = np.random.randint(low=1, high=5, size=len(metadata))
    metadata['duration'] = [librosa.get_duration(path=file) for file in
                            metadata['path']]  # a bit lengthy, but a fraction of the overall training time; maybe I could cache it? Split off a build_metadata function?

    class_durations = []
    for c in classes:
        snip_duration = metadata[metadata['classification'] == c]["duration"]
        total_duration = sum(snip_duration)

        class_durations.append(total_duration)

    weight_raw = []  # gah this is so hacky
    for dur in class_durations:
        weight_raw.append(sum(class_durations) / dur)

    weights = []
    for raw in weight_raw:
        weights.append(raw / sum(weight_raw))

    map_class_to_weight = {index: weight for index, weight in enumerate(weights)}

    map_class_to_id = {name: index for index, name in enumerate(classes)}
    metadata['target_model'] = [map_class_to_id[c] for c in metadata['classification']]
    metadata.to_csv(path_or_buf=os.path.join(dir_model, "metadata.csv"))

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
    ], name=modelName)

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

    model.save(os.path.join("models", modelName), include_optimizer=True)


if __name__ == "__main__":
    if os.path.exists("./models/test"):
        import shutil
        shutil.rmtree("./models/test")
    generate_model("test", 10)
