import os

# File structure
#
DIR_AUDIO = 'audio_in'
DIR_MODELS = 'models'
DIR_EMBEDDERS = 'embedders'

SUBDIR_OUTPUT = 'output'

TRAIN_DIR = 'training'
TRAIN_DIR_AUDIO = os.path.join(TRAIN_DIR, 'audio')
TRAIN_DIR_TESTS = os.path.join(TRAIN_DIR, 'tests')
TRAIN_DIR_TESTS_AUDIO = os.path.join(TRAIN_DIR_TESTS, 'audio')
TRAIN_DIR_TESTS_EMBEDDINGS = os.path.join(TRAIN_DIR_TESTS, 'embeddings')
TRAIN_DIR_TESTS_ANNOTATIONS = os.path.join(TRAIN_DIR_TESTS, 'annotations')

TRAIN_DIR_SET = os.path.join(TRAIN_DIR, 'sets')

TRAIN_DIRNAME_AUDIOSAMPLES = 'samples_audio'
TRAIN_DIRNAME_EMBEDDINGSAMPLES = 'samples_embeddings'

TRAIN_DIRNAME_RAW = 'raw'
TRAIN_DIRNAME_AUGMENT_NOISE = 'augment_noise'
TRAIN_DIRNAME_AUGMENT_VOLUME = 'augment_volume'
TRAIN_DIRNAME_AUGMENT_COMBINE = 'augment_combine'

TRAIN_DIR_AUGMENT = os.path.join(TRAIN_DIR, 'augmentation')

# Files
TAG_EOF = '_finalframe'

# Results
SUFFIX_RESULT_COMPLETE = '_buzzdetect.csv'
SUFFIX_RESULT_PARTIAL = '_buzzpart.csv'
PREFIX_COLUMN_ACTIVATION = 'activation_'
PREFIX_COLUMN_DETECTION = 'detections_'

# Audio
BAD_READ_ALLOWANCE = 0.01  # as a proportion, how much of the tail end of a file can be corrupt without elevating the message to a warning?
# see WorkerStreamer; we often have bad reads at the very end of mp3 audio when the recorder dies during recording; these will be treated as DEBUG reports
FILE_SIZE_MINIMUM = 5000  # files below this size (in bytes) will be skipped (these are often corrupted files that can cause troubles with analysis)

# Analysis
DEFAULT_MODEL = 'model_general_v3'