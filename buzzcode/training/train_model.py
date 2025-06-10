from buzzcode.utils import setthreads

setthreads(8)

import os
import pickle
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from buzzcode import config as cfg
from buzzcode.training.training import build_fold_dataset, clean_name
from buzzcode.training.translation import build_translation_dict


# TODO: re-implement testfold training with new set approach
def train_model(modelname, setname, translationname, epochs_in=300, augment=False):
    dir_model = os.path.join(cfg.DIR_MODELS, modelname)
    if os.path.exists(dir_model) and modelname != 'test':
        raise FileExistsError(
            'a model folder with this name already exists; delete or rename the existing model folder and re-run')
    os.makedirs(dir_model, exist_ok=True)

    translationname = clean_name(translationname, prefix='translation_', extension='.csv')
    translation = pd.read_csv(os.path.join(cfg.DIR_TRAIN_TRANSLATE, 'translation_' + translationname + '.csv'))
    translation.to_csv(os.path.join(dir_model, 'translation.csv'), index=False)

    translation_dict = build_translation_dict(translation)

    classes = set(translation_dict.values())  # TODO: drop np.nan
    classes -= {np.nan}
    classes = list(classes)

    # ---- load data ----
    print('TRAINING: loading data')
    data_train = build_fold_dataset('train', setname, translation_dict, classes, augment=augment)
    data_val = build_fold_dataset('validate', setname, translation_dict, classes, augment=augment)

    # ---- weighting ----
    weights = pd.DataFrame()
    weights['target'] = range(len(classes))
    weights['class'] = classes

    def return_class_samples(class_in, dataset_in):
        class_samples = [s for s in dataset_in if class_in in s['labels_translate']]
        return class_samples

    weights['frames'] = [len(return_class_samples(c, data_train)) for c in weights['class']]
    frames_total = weights['frames'].sum()

    n_classes_present = sum(weights['frames'] > 0)

    def weighter(frames_class):
        if frames_class == 0:
            return 1

        # this weighting seems to over-penalize extremely abundant classes;
        # or maybe it's an effect of augmentation—it would be the perfect penalty,
        # if the frames were IID, but augmentation means that volume outgrows signal
        # so...make an adjustment for augmented data? Penalize less? Hmmmm....
        weight = frames_total / (frames_class * n_classes_present)

        return weight

    weights['weight'] = [weighter(f) for f in weights['frames']]
    weights.to_csv(os.path.join(dir_model, 'weights.csv'), index=False)

    weight_dict = weights.set_index("target")["weight"].to_dict()

    # Turning datasets into tensorflow datasets
    #
    size_batch = 65568  # this is a huge batch, but I'm finding large batches (>1024) descend the gradient much better
    size_shuffle = len(data_train)

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
                                                patience=20,
                                                min_delta=0.01,
                                                restore_best_weights=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 0.001 is default

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

    with open(os.path.join(cfg.DIR_TRAIN_SET, setname, 'config_set.txt'), 'r') as file:
        config_set = json.load(file)

    config_train = {
        'embedder': config_set['embedder'],
        'set': setname,
        'augment': augment,
        'translation': translationname,
        'classes': classes,
        'size_shuffle': size_shuffle,
        'size_batch': size_batch
    }

    with open(os.path.join(dir_model, 'config_model.txt'), 'x') as f:
        f.write(json.dumps(config_train))

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
