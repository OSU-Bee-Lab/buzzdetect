from embedders.yamnet.params import Params
from embedders.yamnet.yamnet import yamnet_frames_model
import keras
import os

dir_embedder = 'embedders/yamnet'

# using frames model and then modifying because we
model = yamnet_frames_model(Params())

# where yamnet.h5 is downloaded from https://storage.googleapis.com/audioset/yamnet.h5
model.load_weights(os.path.join(dir_embedder, 'weights_keras2/yamnet.h5'))

model_activation_only = keras.Model(
    inputs=model.input,
    outputs=model.get_layer("global_average_pooling2d").output,
    name=f"{model.name}_activation_only"
)

# test processing
# import numpy as np
# z = np.zeros(15360)
#
# pred_full = model(z)[1][0]
# pred_act = model_activation_only(z)[0]
# all(pred_full == pred_act)

# save models
# model.save('yamnet_full.keras')
path_out = os.path.join(dir_embedder, 'yamnet.keras')
model_activation_only.save(path_out)

# test loading
# model_loaded = keras.models.load_model(path_out)