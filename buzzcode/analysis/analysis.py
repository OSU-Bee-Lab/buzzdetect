import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import re

import buzzcode.config as cfg


# Functions for handling models
#

def load_model(modelname):
    dir_model = os.path.abspath(os.path.join(cfg.DIR_MODELS, modelname))
    model = tf.keras.models.load_model(dir_model)

    return model


def load_model_config(modelname):
    dir_model = os.path.join('./models', modelname)
    with open(os.path.join(dir_model, 'config_model.txt'), 'r') as cfg:
        config = json.load(cfg)

    if 'embedder' in config.keys():
        print('COMPATIBILITY: updating old embdder config keys')
        config['embeddername'] = config['embedder']
        del config['embedder']

        with open(os.path.join(dir_model, 'config_model.txt'), 'w') as cfg:
            json.dump(config, cfg)

    return config


# Functions for applying transfer model
#

def trim_results(results, classes, framehop_s, digits_time, time_start=0, classes_keep='all', digits_results=2):
    results = np.array(results)
    results = results.round(digits_results)

    classes_out = classes.copy()

    if classes_keep != 'all':
        classes_unknown = set(classes_keep) - set(classes)
        if classes_unknown:
            raise ValueError(f"Bad classes in classes_keep: {', '.join(list(classes_unknown))}")

        keep_indices = [i for i, cls in enumerate(classes) if cls in classes_keep]
        results = results[:, keep_indices]
        classes_out = [classes[i] for i in keep_indices]

    df = pd.DataFrame(results)
    df.columns = classes_out

    df.insert(
        column='start',
        value=range(len(df)),
        loc=0
    )

    df['start'] = df['start'] * framehop_s
    if time_start != 0:
        df['start'] = df['start'] + time_start

    df['start'] = round(df['start'], digits_time)

    return df


def get_framelength_digits(framelength):
    framelength_str = str(framelength)
    framelength_str = re.sub('^\\d+\\.', '', framelength_str)
    return len(framelength_str)