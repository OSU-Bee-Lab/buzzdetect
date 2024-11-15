from buzzcode.utils import save_pickle, load_pickle, setthreads
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import os
import json
setthreads(1)

# modelname = 'test'; setname = 'yamnet_droptraffic_general_semi'; epochs_in=100; test=True
def train_model(modelname, setname, epochs_in=100, test=True):
    dir_model = os.path.join("./models/", modelname)
    if os.path.exists(dir_model) and modelname != 'test':
        raise FileExistsError(
            'a model folder with this name already exists; delete or rename the existing model folder and re-run')
    os.makedirs(dir_model, exist_ok=True)

    dir_set = './training/sets/translated/' + setname
    shutil.copy(os.path.join(dir_set, 'annotations.csv'), os.path.join(dir_model, 'annotations.csv'))

    with open(os.path.join(dir_set, 'config.txt'), 'r') as f:
        config = json.load(f)

    classes = config['classes']

    # Load cache and make dataset
    #
    # 0: inputs; 1:targets; 2:folds_raw; 3:labels
    dataset_all_raw = load_pickle(os.path.join(dir_set, 'data'))

    fold_dict = {1: 'train', 2: 'train', 3: 'train', 4: 'validation', 5: 'test'}

    # input semantic fold, return dataset
    def datasetter(fold_in):
        data_sub = [s for s in dataset_all_raw if fold_dict[s['fold']] == fold_in]
        return data_sub

    dataset_val_raw = datasetter('validation')
    dataset_train_raw = datasetter('train')

    # ADJUSTING TRAINING VOLUME ----
    #
    def count_frames(target_in, set_in):
        targets = [s['targets'][target_in] for s in set_in]
        frames_target = sum(targets)
        return frames_target

    # temporarily turning off evening of dataset; gonna try different conversions first
    # def even_dataset(set_in):
    #     volumes = pd.DataFrame()
    #     volumes['class'] = classes
    #     volumes['target'] = range(len(classes))
    #     volumes['frames'] = [count_frames(t, set_in) for t in volumes['target']]
    #
    #     buzz_frames = volumes.loc[volumes['class'] == 'ins_buzz', 'frames'].item()
    #     volumes['frame_ratio'] = [buzz_frames/f for f in volumes['frames']]  # ratio of buzz frames to train frames
    #
    #     dataset_sub = []
    #     # subset loop
    #     for c in classes:
    #         if volumes.loc[volumes['class'] == c, 'frame_ratio'].item() >= 1:
    #             print(f'class {c} has fewer frames than buzz; skipping')
    #             continue
    #
    #         print(f'subsetting class {c}')
    # TODO: adjust val set also so validation loss isn't biased towards planes

    # WEIGHTING ----
    #
    frames_total = len(dataset_train_raw)

    def weighter(target_in):
        frames_target = count_frames(target_in, dataset_train_raw)
        if frames_target == 0:
            print(f'Warning: no frames found for class {classes[target_in]}')
            return 1

        weight = frames_total/(frames_target*len(classes))  # still overweights huge classes

        return frames_target, weight

    weights = pd.DataFrame()
    weights['target'] = range(len(classes))
    weights['class'] = classes
    weights_frames, weights_weights = zip(*[weighter(c) for c in weights['target']])
    weights['frames'] = weights_frames
    weights['weight'] = weights_weights
    weights.to_csv(os.path.join(dir_model, 'weights.csv'))

    weight_dict = {t: w for t, w in zip(weights['target'], weights['weight'])}  # I guess I could just enumerate

    # Turning datasets into tensorflow datasets
    #
    size_shuffle = 1000
    size_batch = 4096
    config.update(
        {'size_shuffle': size_shuffle, 'size_batch': size_batch}
    )

    def set_to_tfset(set_in):
        inputs, targets = zip(*[(s['embeddings'], s['targets']) for s in set_in])
        inputs = list(inputs)  # TODO: check if necessary for tf.data.Dataset.from_tensor_slices
        targets = list(targets)

        dataset_tf = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset_tf = dataset_tf.cache().shuffle(size_shuffle).batch(size_batch).prefetch(tf.data.AUTOTUNE)

        return dataset_tf

    # These steps take a while, but I think that's unavoidable due to dataset size
    dataset_tf_train = set_to_tfset(dataset_train_raw)
    dataset_tf_val = set_to_tfset(dataset_val_raw)

    # Model creation
    #
    model = tf.keras.Sequential(name=modelname)
    model.add(tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input'))
    model.add(tf.keras.layers.Dense(len(classes)))

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                min_delta=0.01,
                                                restore_best_weights=True)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  # TODO: test the new loss function to see if it's working how I would expect
                  optimizer="adam",
                  metrics=['accuracy'])

    history = model.fit(dataset_tf_train,
                        epochs=epochs_in,
                        validation_data=dataset_tf_val,
                        callbacks=callback,
                        class_weight=weight_dict)

    # TODO: restore weights from where patience ran out, not where val_loss was minimum (worried about overfitting)
    # TODO: then, increase patience
    epoch_stopped = callback.stopped_epoch
    epoch_loss = history.history['val_loss'][epoch_stopped]
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

    plt.annotate(f'stopped at epoch {epoch_stopped} with val_loss: {round(epoch_loss, 3)}', (0,0.05))
    plt.vlines(x=epoch_stopped, ymin=0, ymax=loss_max)

    path_plot = os.path.join(dir_model, 'loss_curves.png')
    plt.savefig(path_plot)
    plt.close()

    save_pickle(os.path.join(dir_model, 'history'), history)
    model.save(os.path.join(dir_model), include_optimizer=True)
    with open(os.path.join(dir_model, 'config.txt'), 'x') as f:
        f.write(json.dumps(config))

    # testing
    #
    # TODO: add analysis of pre-embedded 11m
    # TODO: add affinity to weights
    if test:
        from buzzcode.analyze_audio import translate_results
        path_testfold = os.path.join(dir_model, '/tests/output_testfold.csv')

        dataset_test = datasetter('test')

        inputs_test = [s['embeddings'] for s in dataset_test]
        labels_test = ['; '.join(s['labels_conv']) for s in dataset_test]

        results = model(np.array(inputs_test))
        output = translate_results(np.array(results), classes)
        output.insert(loc=2, column='classes_actual', value=labels_test)
        output.to_csv(path_testfold, index=False)

