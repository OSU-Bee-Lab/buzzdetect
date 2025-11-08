import buzzcode.config as cfg
from buzzcode.analysis.assignments import loglevels
import numbers
import os

class ArgValid:
    def __init__(self, valid, message):
        self.valid = valid
        self.message = message

def validate_modelname(modelname: str):
    modelname = str(modelname)

    if not os.path.exists(os.path.join(cfg.DIR_MODELS, modelname)):
        return ArgValid(False, f'Model folder does not exist for model "{modelname}"')

    if not os.path.exists(os.path.join(cfg.DIR_MODELS, modelname, 'config_model.json')):
        return ArgValid(False, f'Config file does not exist for model "{modelname}"')

    if not os.path.exists(os.path.join(cfg.DIR_MODELS, modelname, 'model.py')):
        return ArgValid(False, f'model.py not found for model "{modelname}"')

    return ArgValid(True, None)


def validate_classes_out(classes_out: list):
    if classes_out == 'all':
        return ArgValid(True, None)

    if classes_out.__class__ is not list:
        return ArgValid(False, 'must be a list')

    if any([i.__class__ is not str for i in classes_out]):
        return ArgValid(False, 'must be a list of strings')

    return ArgValid(True, None)

def validate_precision(precision: float):
    if precision is None:
        return ArgValid(True, None)
    try:
        precision = float(precision)
    except ValueError:
        return ArgValid(False, "must be numeric")
    if precision <= 0:
        return ArgValid(False, "must be > 0")
    if precision < 0.9:
        return ArgValid(True, "analyses with precision < 0.9 are likely to be prone to false positives")
    if precision >= 1:
        return ArgValid(False, 'must be < 1')
    else:
        return ArgValid(True, None)

framehop_prop_warning =    ('Currently, analyses with framehop > 1 will produce valid results, '
        'but buzzdetect will interpret the resulting gaps as missing data.\n'
        f'Fully analyzed files will not be converted from {cfg.SUFFIX_RESULT_PARTIAL} '
        f'to {cfg.SUFFIX_RESULT_COMPLETE}.\n'
        f'Repeated analysis will attempt to fill gaps between frames.')

def validate_framehop(framehop_prop: float):
    try:
        framehop_prop = float(framehop_prop)
    except ValueError:
        return ArgValid(
            False,
            f"must be numeric"
        )
    if framehop_prop <= 0:
        return ArgValid(False, "must be > 0")

    if framehop_prop > 1:
        return ArgValid(True, framehop_prop_warning)
    else:
        return ArgValid(True, None)

def validate_chunklength(chunklength_s: float):
    if type(chunklength_s) is not int:
        try:
            chunklength_s = float(chunklength_s)
        except ValueError:
            return ArgValid(False, f"must be numeric")

    if chunklength_s <= 0:
        return ArgValid(False, "must be > 0")

    return ArgValid(True, None)

def validate_int(value: int, none_ok: bool, value_min: int = None, value_max: int = None):
    if value is None:
        if none_ok:
            return ArgValid(True, None)
        else:
            return ArgValid(False, "cannot be None")

    if type(value) is not int:
        try:
            value = int(value)
        except ValueError:
            return ArgValid(False, "must be an integer")

    if not isinstance(value, numbers.Number):
        return ArgValid(False, "must be numeric")

    if round(value) != value:
        return ArgValid(False, "must be an integer")

    if value_min is not None and value < value_min:
        return ArgValid(False, f"must be >= {value_min}")

    if value_max is not None and value > value_max:
        return ArgValid(False, f"must be <= {value_max}")

    return ArgValid(True, None)

def validate_analyzers_cpu(analyzers_cpu: int):
    return validate_int(
        value=analyzers_cpu,
        value_min=0,
        none_ok=False,
        argname='CPU analyzer count'
    )

def validate_analyzer_gpu(analyzer_gpu: bool):
    return validate_int(
        value=analyzer_gpu,
        value_min=0,
        none_ok=False,
        value_max=1,
        argname='GPU analyzer'
    )

def validate_n_streamers(n_streamers: int):
    return validate_int(
        value=n_streamers,
        value_min=0,
        none_ok=True,
        argname='Number of streamers'
    )


def validate_stream_buffer_depth(stream_buffer_depth: int):
    return validate_int(
        value=stream_buffer_depth,
        value_min=0,
        none_ok=True,
        argname='Stream buffer depth'
    )

def validate_dir_audio(dir_audio: str):
    if not os.path.exists(dir_audio):
        return ArgValid(False, f'folder does not exist')
    return ArgValid(True, None)

def validate_dir_out(dir_out: str):
    if not os.path.exists(dir_out):
        return ArgValid(True, f'Output folder does not exist; it will be created upon analysis')
    return ArgValid(True, None)

def validate_verbosity(verbosity_str: str):
    if verbosity_str not in loglevels.keys():
        return ArgValid(False, f'must be one of: {', '.join(loglevels.keys())}')
    return ArgValid(True, None)

def validate_log_progress(log_progress: bool):
    return validate_int(
        value=log_progress,
        none_ok=False,
        argname='log_progress',
        value_max=1,
    )

validate_map = {
    'modelname': validate_modelname,
    'classes_out': validate_classes_out,
    'precision': validate_precision,
    'framehop_prop': validate_framehop,
    'chunklength': validate_chunklength,
    'analyzers_cpu': validate_analyzers_cpu,
    'analyzer_gpu': validate_analyzer_gpu,
    'n_streamers': validate_n_streamers,
    'stream_buffer_depth': validate_stream_buffer_depth,
    'dir_audio': validate_dir_audio,
    'dir_out': validate_dir_out,
    'verbosity_print': validate_verbosity,
    'verbosity_log': validate_verbosity,
    'log_progress': validate_log_progress,
}