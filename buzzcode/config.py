import os

# File structure
#
DIR_AUDIO = './audio_in'
DIR_MODELS = './models'
DIR_EMBEDDERS = './embedders'

DIR_TRAIN = './training'
DIR_TRAIN_AUDIO = os.path.join(DIR_TRAIN, 'audio')
DIR_TRAIN_ANNOTATION = os.path.join(DIR_TRAIN, 'annotations')
DIR_TRAIN_FOLD = os.path.join(DIR_TRAIN, 'folds')
DIR_TRAIN_SET = os.path.join(DIR_TRAIN, 'sets')
DIR_TRAIN_AUGMENT = os.path.join(DIR_TRAIN, 'augmentation')
DIR_TRAIN_TRANSLATE = os.path.join(DIR_TRAIN, 'translations')

# File tags
TAG_EOF = '_finalframe'
SUFFIX_RESULT_COMPLETE = '_buzzdetect.csv'
SUFFIX_RESULT_PARTIAL = '_buzzpart.csv'
