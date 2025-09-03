import json
import os
import buzzcode.config as cfg

from abc import ABC, abstractmethod
from buzzcode.embedding.load_embedder import load_embedder, BaseEmbedder

class BaseModel(ABC):
    """Abstract base class for all buzzdetect models"""

    # Class attributes that each embedder should define
    modelname: str = None
    embeddername: str = None
    digits_results: int = None  # how many digits should result files be rounded to?
    dtype_in: str = None

    def __init__(self, framehop_prop):
        """Initialize model
        """
        self.model = None
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