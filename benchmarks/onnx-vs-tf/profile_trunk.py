import collections, json, os, sys
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.getcwd())

path = 'embedders/yamnet_onnx/yamnet.onnx'
so = ort.SessionOptions(); so.enable_profiling = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(path, sess_options=so, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
n_in = s.get_inputs()[0].name
print('input', n_in, s.get_inputs()[0].shape, '| providers', s.get_providers())
patches = np.zeros((209, 96, 64), dtype=np.float32)
try:
    s.run(None, {n_in: patches})
except Exception:
    patches = np.zeros((209, 1, 96, 64), dtype=np.float32)
    s.run(None, {n_in: patches})
RUNS = 8
for _ in range(2): s.run(None, {n_in: patches})
for _ in range(RUNS): s.run(None, {n_in: patches})
p = s.end_profiling()
node = collections.defaultdict(lambda: [0.0, None])
for e in json.load(open(p)):
    if e.get('cat') == 'Node' and e['name'].endswith('_kernel_time'):
        k = e['name'][:-len('_kernel_time')]
        node[k][0] += e['dur']/1000.0
        node[k][1] = e.get('args',{}).get('op_name')
tot = sum(v[0] for v in node.values())
print(f'total {tot/RUNS:.2f} ms/run')
print('--- top nodes ---')
for k,v in sorted(node.items(), key=lambda kv:-kv[1][0])[:8]:
    print(f'  {k[:52]:52s} {str(v[1]):10s} {v[0]/RUNS:7.2f} ms  {100*v[0]/tot:5.1f}%')
byop = collections.defaultdict(float)
for k,v in node.items(): byop[v[1]] += v[0]
print('--- by op ---')
for k,t in sorted(byop.items(), key=lambda kv:-kv[1])[:6]:
    print(f'  {str(k):20s} {t/RUNS:7.2f} ms  {100*t/tot:5.1f}%')
os.remove(p)
