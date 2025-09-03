import importlib.util
from pathlib import Path

import buzzcode.config as cfg
from buzzcode.models.BaseModel import BaseModel
from buzzcode.utils import setthreads

setthreads(1)

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

    # Find the model class (should inherit from Basemodel)
    model_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if (isinstance(item, type) and
                issubclass(item, BaseModel) and
                item is not BaseModel):
            model_class = item
            break

    if model_class is None:
        raise ValueError(f"No BaseModel subclass found in {modelname}/model.py")

    # Instantiate and load
    model = model_class(framehop_prop=framehop_prop)

    if initialize:
        model.initialize()

    return model
