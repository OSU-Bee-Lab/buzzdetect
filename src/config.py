import os

# File structure
#
DIR_AUDIO = 'audio_in'

SUBDIR_OUTPUT = 'output'

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

# embedding
DIR_EMBEDDERS = 'embedders'

# models
DIR_MODELS = 'models'
DEFAULT_MODEL = 'model_general_v3'
SUBDIR_TESTS = 'tests'
FNAME_METRICS = 'metrics.csv'