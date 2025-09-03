import importlib.util
from pathlib import Path

import buzzcode.config as cfg
from buzzcode.embedding.BaseEmbedder import BaseEmbedder
from buzzcode.utils import setthreads

setthreads(1)

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
