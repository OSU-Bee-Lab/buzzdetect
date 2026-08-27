"""verify_fusion, but with the CoreML provider options the engine actually ships.

verify_fusion.py asks for CoreMLExecutionProvider bare, which means the default
NeuralNetwork format on ALL compute units -- fp16 on the Neural Engine. The
engine pins ModelFormat=MLProgram, MLComputeUnits=CPUAndGPU to stay in fp32
(src/inference/onnx.py). Those are different graph partitioners, so the
placement question has to be asked of both.
"""
import collections, json, os, statistics, sys, tempfile, time
import numpy as np, onnx, onnxruntime as ort
from fuse_conv_relu import fuse_conv_relu

CONFIGS = {
    'coreml-default (NeuralNetwork, ALL)': ('CoreMLExecutionProvider', {}),
    'coreml-mlprogram-cpugpu (shipped)': ('CoreMLExecutionProvider',
        {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}),
    'coreml-mlprogram-all': ('CoreMLExecutionProvider',
        {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL'}),
    'cpu': ('CPUExecutionProvider', {}),
}

def session(path, ep, opts, profile=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = profile
    if ep == 'CPUExecutionProvider':
        return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    return ort.InferenceSession(path, so, providers=[ep, 'CPUExecutionProvider'],
                                provider_options=[dict(opts), {}])

def placement(path, ep, opts, feeds):
    s = session(path, ep, opts, profile=True)
    for _ in range(2):
        s.run(None, feeds)
    prof = s.end_profiling()
    counts, cpu_ops = collections.Counter(), collections.Counter()
    try:
        for e in json.load(open(prof)):
            if e.get('cat') == 'Node' and e['name'].endswith('_kernel_time'):
                a = e.get('args', {}); p = a.get('provider', '?')
                counts[p] += 1
                if 'CPU' in str(p):
                    cpu_ops[a.get('op_name', '?')] += 1
    finally:
        os.path.exists(prof) and os.remove(prof)
    return counts, cpu_ops

def timeit(path, ep, opts, feeds, repeats=20, warmup=5):
    s = session(path, ep, opts)
    for _ in range(warmup):
        s.run(None, feeds)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); r = s.run(None, feeds); np.asarray(r[0])
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000

def make_feeds(sess, seconds=200):
    rng = np.random.default_rng(0)
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else
                 (seconds * 16000 if len(i.shape) == 1 else 209) for d in i.shape]
        feeds[i.name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return feeds

def main():
    src = sys.argv[1]
    fused = sys.argv[2] if len(sys.argv) > 2 else None
    if fused is None:
        m, n = fuse_conv_relu(onnx.load(src))
        fused = os.path.join(tempfile.mkdtemp(), 'fused.onnx')
        onnx.save(m, fused)
        print(f'fused {n} Conv+Relu pairs\n')
    # reference: CPU EP on the unfused graph, the fp32 ground truth
    ref_sess = session(src, 'CPUExecutionProvider', {})
    feeds = make_feeds(ref_sess)
    ref = ref_sess.run(None, feeds)[0]
    print(f'reference: CPU EP, unfused, output {ref.shape}\n')
    for label, (ep, opts) in CONFIGS.items():
        print(f'--- {label} ---')
        try:
            for arm, path in (('as exported', src), ('fused', fused)):
                out = session(path, ep, opts).run(None, feeds)[0]
                d = float(np.abs(out - ref).max())
                counts, cpu_ops = placement(path, ep, opts, feeds)
                brief = '  '.join(f'{k.replace("ExecutionProvider","")}={v}'
                                  for k, v in counts.most_common())
                t = timeit(path, ep, opts, feeds)
                print(f'  {arm:12s} {t:8.1f} ms   max|d| vs cpu-ref {d:.2e}   {brief}')
                if cpu_ops and ep != 'CPUExecutionProvider':
                    print(f'{"":16s}on CPU: ' + '  '.join(f'{k}={v}' for k, v in cpu_ops.most_common(6)))
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {str(e)[:200]}')
        print()

main()
