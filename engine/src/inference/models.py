import importlib.util
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from src import config as cfg
from src.inference.embedding import BaseEmbedder
from src.inference.embedding import load_embedder


class BaseModel(ABC):
    """Abstract base class for all buzzdetect models"""

    # Class attributes that each embedder should define
    modelname: str = None
    embeddername: str = None
    digits_results: int = None  # how many digits should result files be rounded to?
    dtype_in: str = None
    # Whether this model's inference runs in TensorFlow. Only TF needs its
    # device placement managed by hand (see WorkerInferer._managememory);
    # onnxruntime picks its own execution provider and must not be judged by
    # what TensorFlow can see -- TF reports no GPU on macOS, where onnxruntime
    # has CoreML. Defaults True because every model here but the ONNX ones is
    # a Keras model.
    uses_tensorflow: bool = True

    def __init__(self, framehop_prop):
        """Initialize model
        """
        self.model = None
        # Set by WorkerInferer before initialize(); 'CPU' or 'GPU'.
        self.processor = 'CPU'
        self.embedder: BaseEmbedder = load_embedder(embeddername=self.embeddername, framehop_prop=framehop_prop, initialize=False)

        with open(os.path.join(cfg.DIR_MODELS, self.modelname, 'config_model.json'), 'r') as f:
            self.config = json.load(f)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self, audiosamples):
        """Generate results for audio data"""
        pass


# Which runtime a DualRuntimeModel should prefer: 'onnx', 'tensorflow', or
# 'auto' (the default -- whichever is installed, ONNX first). An environment
# variable rather than an analysis parameter because it changes nothing about
# the results: the two builds are checked against each other at export time and
# agree to float32 rounding. It only changes how fast they arrive, and which
# one wins depends on the machine -- ONNX on a Mac, where it gets CoreML and
# TensorFlow gets no GPU at all; possibly TensorFlow on a Linux box with CUDA.
ENV_RUNTIME = 'BUZZDETECT_RUNTIME'


def _importable(name):
    try:
        __import__(name)
    except ImportError:
        return False
    return True


class DualRuntimeModel(BaseModel):
    """A model whose directory holds both an ONNX graph and Keras weights.

    One directory per model, runnable either way: model.onnx for onnxruntime,
    and either a model.keras or a SavedModel (saved_model.pb + variables/) for
    TensorFlow. Whichever is present and runnable here is used, so the same
    directory serves the frozen sidecar -- which carries no TensorFlow at all
    -- and a checkout that does.

    The choice has to be made in __init__ rather than initialize(), because
    BaseModel.__init__ loads the embedder and the two runtimes need different
    ones: the ONNX YAMNet trunk and the TensorFlow one are separate embedders.
    So `embeddername` and `uses_tensorflow` are instance attributes here rather
    than the class attributes they are on BaseModel.
    """

    embeddername_onnx: str = None
    embeddername_tensorflow: str = None
    # Set when the TensorFlow weights are a SavedModel directory rather than a
    # model.keras file, naming the endpoint to call.
    savedmodel_endpoint: str = None

    def __init__(self, framehop_prop):
        dir_model = os.path.join(cfg.DIR_MODELS, self.modelname)
        self.runtime = self._pick_runtime(dir_model)
        self.uses_tensorflow = self.runtime == 'tensorflow'
        self.embeddername = (self.embeddername_tensorflow if self.uses_tensorflow
                             else self.embeddername_onnx)
        super().__init__(framehop_prop)

    def _has_onnx(self, dir_model):
        return (self.embeddername_onnx is not None
                and os.path.exists(os.path.join(dir_model, 'model.onnx')))

    def _has_tensorflow(self, dir_model):
        return (self.embeddername_tensorflow is not None
                and (os.path.exists(os.path.join(dir_model, 'model.keras'))
                     or os.path.exists(os.path.join(dir_model, 'saved_model.pb'))))

    def _pick_runtime(self, dir_model):
        """Which runtime to run this model under.

        Both the weights in the directory and a runtime installed to run them
        are required, so a model carrying both halves still works in an
        environment that has only one. An explicit request that cannot be met
        is an error rather than a silent fallback: someone who asked for
        TensorFlow wants to know they did not get it.
        """
        want = os.environ.get(ENV_RUNTIME, 'auto').lower()
        available = {
            'onnx': self._has_onnx(dir_model) and _importable('onnxruntime'),
            'tensorflow': self._has_tensorflow(dir_model) and _importable('tensorflow'),
        }
        if want in available:
            if not available[want]:
                raise RuntimeError(
                    f'{ENV_RUNTIME}={want} was requested, but {self.modelname} cannot '
                    f'run under it here: either the weights or the runtime are missing.')
            return want
        if want != 'auto':
            raise ValueError(
                f'{ENV_RUNTIME} must be onnx, tensorflow or auto; got {want!r}')
        # ONNX first: it is what the shipped build carries, and it needs no
        # TensorFlow install to be present.
        for runtime in ('onnx', 'tensorflow'):
            if available[runtime]:
                return runtime
        raise RuntimeError(
            f'{self.modelname} has no weights this environment can run. Looked for '
            f'model.onnx (needs onnxruntime) and model.keras or saved_model.pb '
            f'(needs tensorflow) in {dir_model}.')

    def initialize(self):
        self.embedder.processor = self.processor
        self.embedder.initialize()
        dir_model = os.path.abspath(os.path.join(cfg.DIR_MODELS, self.modelname))
        if self.uses_tensorflow:
            self._initialize_tensorflow(dir_model)
        else:
            self._initialize_onnx(dir_model)

    def _initialize_onnx(self, dir_model):
        from src.inference.onnx import make_session
        self.model = make_session(os.path.join(dir_model, 'model.onnx'), self.processor)
        self.name_in = self.model.get_inputs()[0].name

    def _initialize_tensorflow(self, dir_model):
        path_keras = os.path.join(dir_model, 'model.keras')
        if os.path.exists(path_keras):
            import keras
            self.model = keras.saving.load_model(path_keras, compile=False)
        else:
            from keras.layers import TFSMLayer
            self.model = TFSMLayer(
                dir_model, call_endpoint=self.savedmodel_endpoint or 'serving_default')

    def predict(self, audiosamples):
        return self.predict_embeddings(self.embedder.embed(audiosamples))

    def predict_embeddings(self, embeddings):
        if self.uses_tensorflow:
            results = self.model(embeddings)
            # A SavedModel hands its outputs back in a dict keyed by layer name.
            if isinstance(results, dict):
                (results,) = results.values()
            return results
        import numpy as np
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return self.model.run(None, {self.name_in: embeddings})[0]


def load_model(modelname: str, framehop_prop: float, initialize: bool):
    """
    Generic function to load any model by name.

    Each model directory should contain:
    - model.py: Implementation of BaseModel with class attributes
    """
    model_path = Path(cfg.DIR_MODELS) / modelname

    if not model_path.exists():
        raise ValueError(f"model '{modelname}' not found in {cfg.DIR_MODELS}")

    # Import the model module
    spec = importlib.util.spec_from_file_location(
        f"{modelname}_model",
        model_path / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the model class (should inherit from Basemodel). Match on where the
    # class was defined rather than just on what it subclasses: a model.py that
    # imports a base to subclass -- DualRuntimeModel, say -- would otherwise
    # offer that base as a candidate too, and dir() is alphabetical, so the
    # import would win over the class the file exists to define.
    model_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if (isinstance(item, type) and
                issubclass(item, BaseModel) and
                item.__module__ == module.__name__):
            model_class = item
            break

    if model_class is None:
        raise ValueError(f"No BaseModel subclass defined in {modelname}/model.py")

    # Instantiate and load
    model = model_class(framehop_prop=framehop_prop)

    if initialize:
        model.initialize()

    return model
