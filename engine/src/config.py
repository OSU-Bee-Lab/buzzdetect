import os

# File structure
#
DIR_AUDIO = 'audio_in'

SUBDIR_OUTPUT = 'output'

# Results
SUFFIX_RESULT_COMPLETE = '_buzzdetect.csv'
SUFFIX_RESULT_PARTIAL = '_buzzpart.csv'
PREFIX_COLUMN_ACTIVATION = 'activation_'
PREFIX_COLUMN_DETECTION = 'detections_'

# Audio
DIR_DRIVERS = 'src/stream/drivers'

BAD_READ_ALLOWANCE = 0.01  # as a proportion, how much of the tail end of a file can be corrupt without elevating the message to a warning?
# see WorkerStreamer; we often have bad reads at the very end of mp3 audio when the recorder dies during recording; these will be treated as DEBUG reports
FILE_SIZE_MINIMUM = 5000  # files below this size (in bytes) will be skipped (these are often corrupted files that can cause troubles with analysis)

# models
DIR_MODELS = 'models'
# The desktop app lets users import models without writing into the app bundle;
# it passes their location here as an os.pathsep-separated list of absolute
# paths. The bundled `models/` dir is always searched first.
ENV_MODELS_PATH = 'BUZZDETECT_MODELS_PATH'
DEFAULT_MODEL = 'model_general_v3'
SUBDIR_TESTS = 'tests'
FNAME_METRICS = 'metrics.csv'


def model_roots():
    """Directories to search for model folders, in priority order."""
    roots = [DIR_MODELS]
    for p in os.environ.get(ENV_MODELS_PATH, '').split(os.pathsep):
        if p:
            roots.append(p)
    return roots


def resolve_model_dir(modelname, must_exist=True):
    """Path to a model folder, searched across every root.

    Returns the first root that holds a directory by that name. If none do and
    must_exist is False, returns the path it would have under the bundled root
    (so callers building a default output path don't have to handle the miss);
    otherwise raises FileNotFoundError naming the roots searched.
    """
    for root in model_roots():
        candidate = os.path.join(root, modelname)
        if os.path.isdir(candidate):
            return candidate
    if not must_exist:
        return os.path.join(DIR_MODELS, modelname)
    raise FileNotFoundError(
        f'no model folder "{modelname}" in any of: '
        + ', '.join(os.path.abspath(r) for r in model_roots()))


def list_model_names():
    """Every valid model folder across all roots, deduped and sorted.

    A folder counts if it has a config_model.json. Earlier roots win a name
    collision, so a bundled model shadows an imported one of the same name.
    """
    seen = set()
    for root in model_roots():
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name not in seen and os.path.isfile(
                    os.path.join(root, name, 'config_model.json')):
                seen.add(name)
    return sorted(seen)