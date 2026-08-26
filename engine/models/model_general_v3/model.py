from src.inference.models import DualRuntimeModel


class ModelGeneralV3(DualRuntimeModel):
    """The general buzzdetect model, as an ONNX graph and as TensorFlow weights.

    The TensorFlow half is the original SavedModel this model was trained and
    released as; model.onnx is converted from it. They agree to 5.6e-05 on
    field audio, top class for top class.
    """

    modelname = "model_general_v3"
    embeddername_onnx = 'yamnet_onnx'
    embeddername_tensorflow = 'yamnet_k2'
    digits_results = 2
