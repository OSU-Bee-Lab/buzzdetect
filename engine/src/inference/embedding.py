import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path

from src import config as cfg


class BaseEmbedder(ABC):
    """Abstract base class for all audio embedders"""

    # Class attributes that each embedder should define
    embeddername: str = None
    samplerate: int = None
    framelength_s: float = None  # in seconds
    n_embeddings: int = None
    digits_time: int = None  # how many digits should timestamps be rounded to? Should be equal to framelength digits
    dtype_in: str = None

    def __init__(self, framehop_prop):
        """Initialize embedder with framehop defined as a proportion

        Args:
            framehop_prop: Duration in seconds between successive frames
        """
        self.framehop_prop = framehop_prop
        self.framehop_s = self.framelength_s * framehop_prop
        self.model = None

    @abstractmethod
    def initialize(self):
        """Load embedding model; set load_model=False if you just need attributes"""
        pass

    @abstractmethod
    def embed(self, samples):
        """Generate embeddings for audio data"""
        pass


def load_embedder(embeddername: str, framehop_prop: float, initialize: bool):
    """
    Generic function to load any embedder by name.

    Each embedder directory should contain:
    - embedder.py: Implementation of BaseEmbedder with class attributes
    """
    embedder_path = Path(cfg.DIR_EMBEDDERS) / embeddername

    if not embedder_path.exists():
        raise ValueError(f"Embedder '{embeddername}' not found in {cfg.DIR_EMBEDDERS}")

    # Import the embedder module
    spec = importlib.util.spec_from_file_location(
        f"{embeddername}_embedder",
        embedder_path / "embedder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the embedder class (should inherit from BaseEmbedder)
    embedder_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if (isinstance(item, type) and
                issubclass(item, BaseEmbedder) and
                item is not BaseEmbedder):
            embedder_class = item
            break

    if embedder_class is None:
        raise ValueError(f"No BaseEmbedder subclass found in {embeddername}/embedder.py")

    # Instantiate and load
    embedder = embedder_class(framehop_prop=framehop_prop)

    if initialize:
        embedder.initialize()

    return embedder
