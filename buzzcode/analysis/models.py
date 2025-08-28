import json
import os

import warnings

from buzzcode import config as cfg

def load_model(modelname):
    dir_model = os.path.abspath(os.path.join(cfg.DIR_MODELS, modelname))

    # try loading and catch valueerror
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(dir_model, 'model.keras'), compile=False)
    except ValueError:
        from keras.layers import TFSMLayer
        warnings.warn('Could not load model in .keras format; trying tf SavedModel format with compatibility.')
        model = TFSMLayer(dir_model, call_endpoint='serving_default')

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
