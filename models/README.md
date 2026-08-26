# Bundled models

The ONNX models the desktop app ships with. `scripts/build-engine.mjs` copies
everything here into the app bundle's engine payload, so a model in this
directory is a model users get; anything else stays local.

They're produced from a trained buzzdetect model by:

```
cd engine
.venv/bin/python3 tools/onnxify_model.py <modelname>     # writes engine/models/<modelname>_onnx
```

and then copied here. `engine/models/` stays what it always was: whatever
models you happen to have locally, TensorFlow and ONNX alike, used when running
the engine directly or when the app falls back to a checkout.

Weights are small (a classifier head over YAMNet embeddings, well under a
megabyte). The 12.8 MB YAMNet trunk they share is not here -- it lives once, in
`engine/embedders/yamnet_onnx/yamnet.onnx`.
