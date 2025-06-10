import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf


# Functions for handling models
#

def load_model(modelname):
    dir_model = os.path.join('./models', modelname)
    model = tf.keras.models.load_model(dir_model)

    return model


def load_model_config(modelname):
    dir_model = os.path.join('./models', modelname)
    with open(os.path.join(dir_model, 'config.txt'), 'r') as cfg:
        config = json.load(cfg)

    return config


# Functions for applying transfer model
#

def trim_results(results, classes, classes_keep='all', digits=2):
    # as I move from DataFrames to dicts, this function is becoming less useful...
    results = np.array(results)
    results = results.round(digits)

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

    return df


