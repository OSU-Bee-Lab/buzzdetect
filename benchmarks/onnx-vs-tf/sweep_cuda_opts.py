"""Does tuning the CUDA provider sweep away the two pathological convs?

providers_for() passes {} today, so onnxruntime uses defaults for cuDNN
algorithm selection and keeps NCHW layout. Depthwise convolutions are the
classic case where both choices are wrong.
"""
import itertools, os, statistics, sys, time
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.getcwd())

MODEL = 'models/model_general_v3_onnx/model_combined.onnx'
audio = np.zeros(int(200.0 * 16000), dtype=np.float32)

CONFIGS = [
    ('default (what ships)',            {}),
    ('algo=EXHAUSTIVE',                 {'cudnn_conv_algo_search': 'EXHAUSTIVE'}),
    ('algo=HEURISTIC',                  {'cudnn_conv_algo_search': 'HEURISTIC'}),
    ('max_workspace',                   {'cudnn_conv_use_max_workspace': '1'}),
    ('prefer_nhwc',                     {'prefer_nhwc': '1'}),
    ('nhwc + EXHAUSTIVE',               {'prefer_nhwc': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE'}),
    ('nhwc + EXHAUSTIVE + workspace',   {'prefer_nhwc': '1', 'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                         'cudnn_conv_use_max_workspace': '1'}),
]

print(f'{"config":34s} {"median ms":>10s} {"min ms":>9s}  {"vs default":>10s}')
base = None
for label, opts in CONFIGS:
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(MODEL, sess_options=so,
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                    provider_options=[opts, {}])
        n_in = sess.get_inputs()[0].name
        for _ in range(5):
            sess.run(None, {n_in: audio})
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); sess.run(None, {n_in: audio}); ts.append(time.perf_counter() - t0)
        med = statistics.median(ts) * 1000
        if base is None:
            base = med
        print(f'{label:34s} {med:10.1f} {min(ts)*1000:9.1f}  {base/med:9.2f}x')
        del sess
    except Exception as e:
        print(f'{label:34s}   FAILED: {str(e)[:60]}')
