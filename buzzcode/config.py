import os

dir_models = './models'
dir_embedders = './embedders'

# Training
dir_training = './training'
dir_trainingaudio = os.path.join(dir_training, 'audio')

dir_annotations = os.path.join(dir_training, 'annotations')
dir_folds = os.path.join(dir_training, 'folds')
dir_sets = os.path.join(dir_training, 'sets')
dir_augmentation = os.path.join(dir_training, 'augmentation')
dir_translations = os.path.join(dir_training, 'translations')

# Analysis
dir_audio_in = './audio_in'

# tags
tag_eof = '_finalframe'
