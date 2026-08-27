"""Is the conv pathology a function of batch size?

Everything measured so far runs 209 frames at once (200s at framehop 1). If
cuDNN picks a bad kernel at that shape, --chunklength is a user-facing knob
that changes it.
"""
import os, statistics, sys, time
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.getcwd())

MODEL = 'embedders/yamnet_onnx/yamnet.onnx'
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(MODEL, sess_options=so,
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
n_in = sess.get_inputs()[0].name
print('providers:', sess.get_providers(), '| input', sess.get_inputs()[0].shape)

print(f'\n{"frames":>7s} {"total ms":>9s} {"ms/frame":>9s} {"audio-s/wall-s":>15s} {"vs 209":>8s}')
base = None
for n in (1, 8, 16, 32, 64, 96, 128, 209, 256, 418, 836):
    x = np.zeros((n, 96, 64), dtype=np.float32)
    try:
        for _ in range(3):
            sess.run(None, {n_in: x})
        ts = []
        for _ in range(12):
            t0 = time.perf_counter(); sess.run(None, {n_in: x}); ts.append(time.perf_counter() - t0)
        med = statistics.median(ts) * 1000
        per = med / n
        rate = (n * 0.96) / (med / 1000)     # each frame is 0.96s of audio
        if n == 209:
            base = per
        print(f'{n:7d} {med:9.2f} {per:9.4f} {rate:15.1f}', end='')
        print(f' {base/per:7.2f}x' if base else '')
    except Exception as e:
        print(f'{n:7d}   FAILED: {str(e)[:50]}')

if base:
    print(f'\n(209 frames = 200s at framehop 1, which is --chunklength 200)')
