import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
from buzzcode.tools import save_pickle


# modelname = 'test'; metadata_name="metadata_raw"; weights_name=None; epochs_in=3
def generate_model(modelname, metadata_name="metadata_raw", weights_name=None, epochs_in=100):
    dir_model = os.path.join("./models/", modelname)
    if os.path.exists(dir_model) and modelname != 'test':
        raise FileExistsError('a model folder with this name already exists; delete or rename the existing model folder and re-run')
    os.makedirs(dir_model, exist_ok=True)

    dir_training = ''
    dir_metadata = os.path.join(dir_training, 'metadata')
    dir_embeddings = os.path.join(dir_training, 'embeddings')
    dir_audio = os.path.join(dir_training, 'audio')

    # Acquiring and filtering training data
    #
    if not bool(re.search('\\.csv$',metadata_name)):
        metadata_name = metadata_name + '.csv'
    metadata = pd.read_csv(os.path.join(dir_metadata, metadata_name))

    # trim leading slash, if present, from idents for os.path.join
    metadata['path_relative'] = [re.sub('^/', '', i) for i in metadata['path_relative']]

    if 'path_audio' not in metadata.columns:
        metadata['path_audio'] = [os.path.join(dir_audio, p) for p in metadata['path_relative']]

    if 'fold' not in metadata.columns:
        metadata['fold'] = np.random.randint(low=1, high=6, size=len(metadata))

    classes = metadata['classification'].unique()

    if weights_name is None:
        weightdf = pd.DataFrame()
        weightdf['classification'] = metadata['classification'].unique()
        weightdf['weight'] = 1
        classes = weightdf['classification'].to_list()
    else:
        if not bool(re.search('\\.csv$', weights_name)):
            weights_name = weights_name + '.csv'

        weights_path = os.path.join(dir_training, 'weights', weights_name)

        weightdf = pd.read_csv(weights_path)
        weights_new = pd.DataFrame()

        # if a weight isn't assigned to a class, assign it 1
        weights_new['classification'] = [c for c in classes if c not in weightdf['classification'].values]
        weights_new['weight'] = 1

        weightdf = pd.concat([weightdf, weights_new])

    weightdf = weightdf.sort_values(by='classification')
    weightdf.to_csv(os.path.join(dir_model, "weights.csv"), index=False)

    dict_weight = {index: weight for index, weight in enumerate(weightdf['weight'])}
    dict_names = {name: index for index, name in enumerate(weightdf['classification'])}

    metadata['target'] = [dict_names[c] for c in metadata['classification']]

    metadata['path_embeddings'] = [re.sub(dir_audio, dir_embeddings, path) for path in metadata['path_audio']]
    metadata['path_embeddings'] = [os.path.splitext(emb)[0] + '.npy' for emb in metadata['path_embeddings']]

    metadata = metadata[[os.path.exists(e) for e in metadata['path_embeddings']]]

    metadata.to_csv(os.path.join(dir_model, "metadata.csv"), index=False)

    # dataset creation
    #
    # dataset_full = tf.data.Dataset.from_tensor_slices((metadata['path_audio'], metadata['target'], metadata['fold']))



    dataset_full = tf.data.Dataset.from_tensor_slices((metadata['path_embeddings'], metadata['target'], metadata['fold']))


    def load_embed(path_embeddings, target, fold):
        embed_array = tf.numpy_function(
            func=lambda y: np.array(np.load(y.decode("utf-8"), allow_pickle=True)),
            inp=[path_embeddings],
            Tout=tf.float32,
        )

        embeddings = tf.convert_to_tensor(embed_array)
        n_embeddings = tf.shape(embeddings)[0]

        return embeddings, tf.repeat(target, n_embeddings), tf.repeat(fold, n_embeddings)

    dataset_full = dataset_full.map(lambda paths, labels, folds: load_embed(paths, labels, folds)).unbatch()

    def set_shape(embeddings, target, fold):
        embeddings.set_shape(1024)

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

    history = model.fit(dataset_train,
                        epochs=epochs_in,  # would rather solely rely on early stopping
                        validation_data=dataset_val,
                        callbacks=callback,
                        class_weight=dict_weight)

    save_pickle(os.path.join(dir_model, 'history'), history)

    model.save(os.path.join(dir_model), include_optimizer=True)

if __name__ == "__main__":
    generate_model('test', metadata_name="metadata_raw.csv", epochs_in=1)


