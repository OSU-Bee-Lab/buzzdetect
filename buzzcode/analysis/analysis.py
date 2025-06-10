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

def translate_results(results, classes, digits=2):
    # as I move from DataFrames to dicts, this function is becoming less useful...
    results = np.array(results)
    results = results.round(digits)
    translate = []
    for i, scores in enumerate(results):
        results_frame = {classes[i]: scores[i] for i in range(len(classes))}
        translate.append(results_frame)
    output_df = pd.DataFrame(translate)

    return output_df


