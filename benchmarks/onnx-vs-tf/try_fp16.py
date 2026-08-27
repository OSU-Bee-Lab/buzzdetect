"""Does fp16 move the convolutions?

They are 73.6% of the fused graph and the last untested lever on their
throughput. Turing does packed fp16, though TU117 (this card) may not get the
2:1 rate the bigger Turing dies do -- which is exactly why this is measured
rather than assumed.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnxconverter_common import float16
sys.path.insert(0, os.getcwd())

SRC = 'embedders/yamnet_onnx/yamnet.onnx'
DST = '/tmp/claude-1001/-home-luke-projects-buzzdetect/57b3b6a5-c9a9-4ece-8a37-6c11d8a02630/scratchpad/yamnet_fp16.onnx'

if not os.path.exists(DST):
    m = onnx.load(SRC)
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), DST)
    print('converted to fp16')

def bench(path, label):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, sess_options=so,
                             providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    n_in = s.get_inputs()[0].name
    x = np.zeros((209, 96, 64), dtype=np.float32)
    for _ in range(5): s.run(None, {n_in: x})
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); out = s.run(None, {n_in: x}); ts.append(time.perf_counter()-t0)
    med = statistics.median(ts)*1000
    print(f'{label:24s} {med:8.2f} ms   {(209*0.96)/(med/1000):8.1f} audio-s/wall-s')
    return med, np.asarray(out[0], dtype=np.float64)

f32, out32 = bench(SRC, 'fp32 (current)')
f16, out16 = bench(DST, 'fp16')
print(f'\nspeedup: {f32/f16:.2f}x')
d = np.abs(out32 - out16)
print(f'embedding max abs diff: {d.max():.3g}   mean {d.mean():.3g}')
print(f'(embeddings are ~{np.abs(out32).mean():.3g} in magnitude)')
