import shutil

from buzzcode.utils import setthreads

setthreads(8)

import os
import pickle
import json

import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

from buzzcode import config as cfg
from buzzcode.training.training import clean_name, build_weights, build_classes, can_write_model
from buzzcode.training.training_dataset import build_fold_dataset, load_augment_noise, load_augment_volume


def train_model(modelname, setname, name_translation, name_noise=None, name_volume=None, epochs_in=300):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    if not can_write_model(modelname):
        print('a model folder with this name already exists; delete or rename the existing model folder and re-run')
        return False
    os.makedirs(dir_model, exist_ok=True)

    # ---- load data ----
    print('TRAINING: loading data')
    name_translation = clean_name(name_translation, prefix='translation_', extension='.csv')

    translation = pd.read_csv(os.path.join(cfg.TRAIN_DIR_SET, setname, 'translation_' + name_translation + '.csv'))

    data_train = build_fold_dataset(setname=setname, fold='train', translation=translation, augmenttype='raw')

    if name_noise is not None:
        name_noise = clean_name(name_noise, prefix='augment_noise_', extension='.csv')
        data_train += load_augment_noise(setname=setname, translation=translation, name_noise=name_noise)

    if name_volume is not None:
        name_volume = clean_name(name_volume, prefix='augment_volume_', extension='.csv')
        data_train += load_augment_volume(setname=setname, translation=translation, name_volume=name_volume)

    labels_buzz = translation['from'][translation['to']=='ins_buzz'].to_list()
    data_val = build_fold_dataset(setname=setname, fold='validate', translation=translation, labels_keep_raw=labels_buzz, exclusive=False, augmenttype='raw')

    # ---- weighting ----
    classes = build_classes(translation)
    weights = build_weights(data_train, classes)
    weight_dict = {target: weight for target, weight in enumerate(weights['weight'])}  # for training

    # Turning datasets into tensorflow datasets
    #
    size_batch = 65568
    size_shuffle = 10*size_batch

    def data_to_tfset(set_in):
        embeddings = [s['embeddings'] for s in set_in]
        targets = [s['targets'] for s in set_in]

        dataset_tf = tf.data.Dataset.from_tensor_slices((embeddings, targets))
        dataset_tf = dataset_tf.cache().shuffle(size_shuffle).batch(size_batch).prefetch(tf.data.AUTOTUNE)

        return dataset_tf

    # These steps take a while, but I think that's unavoidable due to dataset size
    print('TRAINING: translating data to tensors')
    data_train_tf = data_to_tfset(data_train)
    data_val_tf = data_to_tfset(data_val)

    # Model creation
    #
    model = tf.keras.Sequential(name=modelname)
    model.add(tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dense(len(classes)))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=30,
                                                min_delta=0.01,
                                                restore_best_weights=True)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001*2)  # 0.001 is default

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(data_train_tf,
                        epochs=epochs_in,
                        validation_data=data_val_tf,
                        callbacks=callback,
                        class_weight=weight_dict)

    with open(os.path.join(dir_model, 'history.pickle'), 'wb') as file:
        pickle.dump(history, file)

    model.save(os.path.join(dir_model), include_optimizer=True)

    # write metadata (after successful training, just in case)
    # copy set info, because sets may change over time (e.g., standard will be updated as new annotations come in)
    dir_set = os.path.join(cfg.TRAIN_DIR_SET, setname)
    shutil.copy(os.path.join(dir_set, 'config_set.txt'), dir_model)
    shutil.copy(os.path.join(dir_set, 'annotations.csv'), dir_model)
    shutil.copy(os.path.join(dir_set, 'folds.csv'), dir_model)
    weights.to_csv(os.path.join(dir_model, 'weights.csv'), index=False)
    translation.to_csv(os.path.join(dir_model, 'translation.csv'), index=False)

    with open(os.path.join(dir_set, 'config_embedder.txt'), 'r') as file:
        config_embedder = json.load(file)
    config_model = {
        'embeddername': config_embedder['embeddername'],
        'set': setname,
        'translation': name_translation,
        'classes': classes,
        'size_shuffle': size_shuffle,
        'size_batch': size_batch
    }
    with open(os.path.join(dir_model, 'config_model.txt'), 'x') as f:
        f.write(json.dumps(config_model))

    epoch_restored = callback.best_epoch
    epoch_loss = history.history['val_loss'][epoch_restored]
    loss_max = max(history.history['val_loss'])

    plt.plot()
    ax = plt.gca()
    ax.set_ylim([0, loss_max])
    plt.title(f'training loss curves for model {modelname}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'], loc='upper left')

    plt.annotate(f'stopped at epoch {epoch_restored} with val_loss: {round(epoch_loss, 3)}', (0, 0.05))
    plt.vlines(x=epoch_restored, ymin=0, ymax=loss_max)

    path_plot = os.path.join(dir_model, 'loss_curves.svg')
    plt.savefig(path_plot)
    plt.close()

    return True


if __name__ == '__main__':
    train_model(modelname='lite', setname='lite', name_translation='general')