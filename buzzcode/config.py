import os

# File structure
#
DIR_AUDIO = 'audio_in'
DIR_MODELS = 'models'
DIR_EMBEDDERS = 'embedders'

TRAIN_DIR = 'training'
TRAIN_DIR_AUDIO = os.path.join(TRAIN_DIR, 'audio')
TRAIN_DIR_TESTS = os.path.join(TRAIN_DIR, 'tests')
TRAIN_DIR_TESTS_AUDIO = os.path.join(TRAIN_DIR_TESTS, 'audio')
TRAIN_DIR_TESTS_EMBEDDINGS = os.path.join(TRAIN_DIR_TESTS, 'embeddings')

TRAIN_DIR_SET = os.path.join(TRAIN_DIR, 'sets')

TRAIN_DIRNAME_AUDIOSAMPLES = 'samples_audio'
TRAIN_DIRNAME_EMBEDDINGSAMPLES = 'samples_embeddings'

TRAIN_DIRNAME_RAW = 'raw'
TRAIN_DIRNAME_AUGMENT_NOISE = 'augment_noise'
TRAIN_DIRNAME_AUGMENT_VOLUME = 'augment_volume'
TRAIN_DIRNAME_AUGMENT_COMBINE = 'augment_combine'

TRAIN_DIR_AUGMENT = os.path.join(TRAIN_DIR, 'augmentation')

# File tags
TAG_EOF = '_finalframe'
SUFFIX_RESULT_COMPLETE = '_buzzdetect.csv'
SUFFIX_RESULT_PARTIAL = '_buzzpart.csv'
