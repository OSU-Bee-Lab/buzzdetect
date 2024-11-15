import os

# frame hop expressed as a multiple of frame length
framehop = 0.5

dir_training = './training'
dir_training_embeddings = os.path.join(dir_training, 'embeddings')

dir_audio_in = './audio_in'
dir_audio_in_embeddings = './audio_embeddings'

dir_models = './models'
dir_model_archive = os.path.join(dir_models, 'archive')