import json
import os

import numpy as np

from src import config as cfg


# Everything a model needs to declare, and the JSON type each value takes.
# buzzdetect-training's tools/export_onnx.py writes these into config_model.json
# straight off the Keras model it exports.
REQUIRED_CONFIG_KEYS = {
    'classes': list,          # output class names, in the graph's column order
    'samplerate': int,        # what the front end expects the waveform at
    'framelength_s': (int, float),   # seconds of audio per frame
    'digits_time': int,       # rounding for the timestamp columns
    'digits_results': int,    # rounding for the activation columns
    'samples_hop': int,       # samples between the start of one frame and the next
    'samples_min': int,       # fewest samples that produce a frame (STFT overhang)
}


def _validate_config(modelname, config):
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(
            f'{modelname}/config_model.json is missing required key(s): '
            f'{", ".join(missing)}. A model exported before config_model.json '
            f'carried the framing parameters needs re-exporting with '
            f'buzzdetect-training tools/export_onnx.py.')
    for key, types in REQUIRED_CONFIG_KEYS.items():
        value = config[key]
        # bool is an int subclass; none of these keys wants one.
        if isinstance(value, bool) or not isinstance(value, types):
            raise ValueError(
                f'{modelname}/config_model.json: "{key}" must be '
                f'{getattr(types, "__name__", types)}, got {value!r}')
    if not config['classes'] or not all(
            isinstance(c, str) for c in config['classes']):
        raise ValueError(
            f'{modelname}/config_model.json: "classes" must be a non-empty '
            f'list of strings')


class OnnxModel:
    """A buzzdetect model: one ONNX graph, waveform in, predictions out.

    Everything the model does is in `model.onnx` -- the log-mel front end, the
    embedding trunk and the classifier head, exported and fused into a single
    graph by buzzdetect-training's `tools/export_onnx.py`. There is no embedder
    plugin and no TensorFlow. A model directory is `model.onnx`, an optional
    `model.fp16.onnx`, and a `config_model.json` holding the class list and the
    framing parameters (see REQUIRED_CONFIG_KEYS). Nothing in the directory is
    executable -- a model is data.

    The session is built at one fixed input length (`samples_session`), which
    the analyzer sets from the chunk length before initialize(). That is not a
    tuning choice: CoreML's MLProgram format cannot compile a graph with an
    unbounded dimension, so a dynamic session simply fails to run on macOS.
    predict() zero-pads each chunk up to that length and returns only the
    frames the real audio covers.
    """

    def __init__(self, modelname, model_dir, config, framehop_prop):
        _validate_config(modelname, config)

        self.modelname = modelname
        self.model_dir = model_dir
        self.config = config

        self.samplerate = config['samplerate']
        self.framelength_s = float(config['framelength_s'])
        self.digits_time = config['digits_time']
        self.digits_results = config['digits_results']
        # Samples between the start of one frame and the next.
        self.samples_hop = config['samples_hop']
        # The fewest samples that produce a frame -- shorter input is padded up
        # to it. Larger than samples_hop, because the front end's STFT window
        # overhangs the frame it belongs to.
        self.samples_min = config['samples_min']

        # The patch hop is welded to the patch window inside the exported
        # graph, so an overlapping framehop cannot be honoured. Refuse it
        # rather than silently analysing on contiguous frames.
        if framehop_prop != 1:
            raise ValueError(
                f'{self.modelname} only supports framehop_prop=1, got '
                f'{framehop_prop}. Overlapping frames would have to be a '
                f'parameter of the exported graph, and are not.')
        self.framehop_prop = framehop_prop
        self.framehop_s = self.framelength_s * framehop_prop

        self.model = None
        # Both set by WorkerInferer before initialize().
        self.processor = 'CPU'
        self.samples_session = None

    def session_length(self, chunklength_s):
        """Samples to build the session for, given the analyzer's chunk length.

        One frame longer than the chunk needs. Resampling a chunk to the
        model's rate does not always land on exactly the expected sample count,
        and a chunk that came back a few samples long would otherwise not fit
        the session it was sized for. The spare frame costs one frame of
        compute per chunk and removes the whole class of problem.
        """
        n_chunk = int(round(chunklength_s * self.samplerate))
        return self.samples_min + self.samples_hop * self.n_frames(n_chunk)

    def n_frames(self, n_samples):
        """How many frames the graph returns for n_samples of audio.

        Two things here are not what they look like.

        The first frame needs more samples than the hop -- the front end pads
        up to a whole patch plus the STFT window's overhang -- so this is not
        ceil(n_samples / samples_hop).

        And the hop division is a float32 multiply by the reciprocal of the
        hop, not a division. tf2onnx emits it that way, and it matters: 1/15360
        is not exact in float32, so at some exact multiples of the hop the
        quotient lands just above the integer and the ceil returns one more
        frame than real arithmetic would. 61680 samples and 3210480 samples are
        two such lengths. Doing this in float64 is wrong by one frame there,
        which would silently drop a frame of audio off the end of a chunk.

        buzzdetect-training's export tool checks this against the graph at
        every exact multiple of the hop before it ships a model.
        """
        if n_samples <= 0:
            return 0
        after = np.float32(max(0, n_samples - self.samples_min))
        reciprocal = np.float32(1.0) / np.float32(self.samples_hop)
        return 1 + int(np.ceil(after * reciprocal))

    def initialize(self):
        from src.inference.onnx import make_session

        if self.samples_session is None:
            raise RuntimeError(
                f'{self.modelname}.samples_session was not set before '
                f'initialize(); the session needs a fixed input length.')
        path = os.path.join(os.path.abspath(self.model_dir), 'model.onnx')
        self.model = make_session(path, self.processor, self.samples_session)
        self.name_in = self.model.get_inputs()[0].name

    def predict(self, audiosamples):
        """Predictions for one chunk of audio at self.samplerate.

        The chunk is padded up to the session's length and the surplus frames
        are dropped here, so nothing downstream ever sees a result computed
        from padding.
        """
        samples = np.asarray(audiosamples, dtype=np.float32)
        n = samples.shape[0]
        if n > self.samples_session:
            raise ValueError(
                f'{self.modelname} got a {n}-sample chunk but its session is '
                f'built for {self.samples_session}. The analyzer sizes the '
                f'session from chunklength; a chunk longer than that should '
                f'not exist.')

        padded = np.zeros(self.samples_session, dtype=np.float32)
        padded[:n] = samples
        results = self.model.run(None, {self.name_in: padded})[0]
        return results[:self.n_frames(n)]


def load_model(modelname: str, framehop_prop: float, initialize: bool):
    """Load a model by name, searched across every model root.

    A model directory is `model.onnx` and a `config_model.json` (see
    REQUIRED_CONFIG_KEYS). The name is the folder name; the folder can live in
    the bundled `models/` dir or in any root the desktop app contributes via
    BUZZDETECT_MODELS_PATH.
    """
    try:
        model_dir = cfg.resolve_model_dir(modelname)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e

    config_path = os.path.join(model_dir, 'config_model.json')
    if not os.path.isfile(config_path):
        raise ValueError(
            f"model '{modelname}' has no config_model.json in {model_dir}")
    with open(config_path) as f:
        config = json.load(f)

    model = OnnxModel(modelname, model_dir, config, framehop_prop=framehop_prop)

    if initialize:
        model.initialize()

    return model
