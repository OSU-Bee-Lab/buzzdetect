"""Is the dynamic batch dimension what makes the two convs slow?

isolate_convs.py times layer 1 at 0.48 ms and layer 2 at 0.87 ms as standalone
graphs with a *static* [209,...] input.  Inside the trunk, whose batch dim is
the symbolic `unk__360`, the profiler charges those same two nodes 18.0 and
28.5 ms.  The only difference left is the dynamic dimension.

Two ways to remove it, neither of which touches the engine:
  - a free-dimension override on the session (no re-export, ships as an option)
  - rewriting the dim to a literal in the graph (what a re-export would give)
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort

sys.path.insert(0, os.getcwd())
MODEL = 'embedders/yamnet_onnx/yamnet.onnx'
N = 209


def timeit(sess, name, runs=30):
    x = ort.OrtValue.ortvalue_from_numpy(np.zeros((N, 96, 64), dtype=np.float32), 'cuda', 0)
    b = sess.io_binding()
    b.bind_ortvalue_input(name, x)
    for o in sess.get_outputs():
        b.bind_output(o.name, 'cuda', 0)
    for _ in range(10):
        sess.run_with_iobinding(b)
    b.synchronize_outputs()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run_with_iobinding(b)
        b.synchronize_outputs()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def make(so=None):
    so = so or ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(MODEL, sess_options=so,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


s = make()
sym = s.get_inputs()[0].shape[0]
print(f'input {s.get_inputs()[0].name} {s.get_inputs()[0].shape}  (symbolic batch dim: {sym!r})')
base = timeit(s, s.get_inputs()[0].name)
print(f'  dynamic batch (as exported)          {base:8.2f} ms   1.00x')

so = ort.SessionOptions()
so.add_free_dimension_override_by_name(sym, N)
s2 = make(so)
t = timeit(s2, s2.get_inputs()[0].name)
print(f'  free_dimension_override -> {N}         {t:8.2f} ms   {base / t:.2f}x')

# and the same thing baked into the graph, as a re-export would produce
m = onnx.load(MODEL)
m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = N
for vi in list(m.graph.value_info) + list(m.graph.output):
    d = vi.type.tensor_type.shape.dim
    if len(d) and d[0].HasField('dim_param') and d[0].dim_param == sym:
        d[0].dim_value = N
onnx.save(m, '/tmp/yamnet_static.onnx')
MODEL = '/tmp/yamnet_static.onnx'
s3 = make()
t3 = timeit(s3, s3.get_inputs()[0].name)
print(f'  batch baked into the graph           {t3:8.2f} ms   {base / t3:.2f}x')
