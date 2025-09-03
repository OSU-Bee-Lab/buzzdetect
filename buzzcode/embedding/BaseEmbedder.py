from abc import ABC, abstractmethod


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