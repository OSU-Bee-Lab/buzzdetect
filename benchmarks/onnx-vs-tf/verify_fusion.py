"""Verify a Conv+Relu fusion: parity, EP placement, and speed.

Standalone -- needs only onnx, onnxruntime and numpy. Nothing from buzzdetect.

    python verify_fusion.py model.onnx                    # fuses to a temp copy
    python verify_fusion.py model.onnx model_fused.onnx   # compares two files
    python verify_fusion.py model.onnx --ep CoreMLExecutionProvider

For each available execution provider it reports three things, in the order
they matter:

  1. PARITY -- must be bit-exact (max abs diff 0.0). The rewrite changes which
     kernel applies the max(), not the arithmetic. Anything nonzero means the
     rewrite is wrong, not that tolerance needs loosening.

  2. PLACEMENT -- how many nodes each EP actually took. This is the whole
     question on CoreML: FusedConv is a com.microsoft contrib op, and if
     CoreML declines it, onnxruntime partitions the graph and hands those
     nodes to the CPU. A graph that was one CoreML partition can become
     dozens of fragments with a copy at every boundary, which is slower than
     never fusing at all. A speedup number alone will not show you this.

  3. TIME -- median wall clock, after warmup.

Note ort.get_available_providers() reports what onnxruntime was *compiled*
with, not what this machine can run. The placement table is the ground truth:
if you ask for CoreML and every node lands on CPU, CoreML did not run.
"""

import argparse
import collections
import json
import os
import statistics
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort

from fuse_conv_relu import fuse_conv_relu

DEFAULT_BATCH = 209          # one 200 s buzzdetect chunk, for a YAMNet trunk
DEFAULT_AUDIO = 200 * 16000  # one 200 s chunk, for a fused waveform-in graph


def make_input(sess, batch, seed=0):
    """Random input matching the graph's declared shape, dynamic dims filled."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for i in sess.get_inputs():
        shape = []
        for d in i.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            elif len(i.shape) == 1:
                shape.append(DEFAULT_AUDIO)   # raw waveform input
            else:
                shape.append(batch)
        feeds[i.name] = rng.standard_normal(shape, dtype=np.float32)
    return feeds


def session(path, ep, profile=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = profile
    providers = [ep] if ep == 'CPUExecutionProvider' else [ep, 'CPUExecutionProvider']
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def placement(path, ep, feeds):
    """Which EP actually executed each node."""
    s = session(path, ep, profile=True)
    for _ in range(2):
        s.run(None, feeds)
    prof = s.end_profiling()
    counts = collections.Counter()
    ops_on_cpu = collections.Counter()
    try:
        for e in json.load(open(prof)):
            if e.get('cat') == 'Node' and e['name'].endswith('_kernel_time'):
                a = e.get('args', {})
                p = a.get('provider', '?')
                counts[p] += 1
                if 'CPU' in str(p):
                    ops_on_cpu[a.get('op_name', '?')] += 1
    finally:
        os.path.exists(prof) and os.remove(prof)
    # counts are node-executions across the profiled runs, not distinct nodes;
    # only the ratio between the two graphs matters here
    return counts, ops_on_cpu


def timeit(path, ep, feeds, repeats=20, warmup=5):
    s = session(path, ep)
    for _ in range(warmup):
        s.run(None, feeds)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = s.run(None, feeds)
        np.asarray(r[0])
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='the unfused .onnx')
    ap.add_argument('fused', nargs='?', help='the fused .onnx (default: build one)')
    ap.add_argument('--ep', action='append', default=None,
                    help='execution provider to test (repeatable; default: all available)')
    ap.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    ap.add_argument('--repeats', type=int, default=20)
    args = ap.parse_args()

    fused_path = args.fused
    if fused_path is None:
        m, n = fuse_conv_relu(onnx.load(args.src))
        fused_path = os.path.join(tempfile.mkdtemp(), 'fused.onnx')
        onnx.save(m, fused_path)
        print(f'fused {n} Conv+Relu pairs -> {fused_path}\n')

    available = ort.get_available_providers()
    # TensorRT needs libraries that are not part of this question, and asking
    # for it when they are absent prints a wall of error text before falling
    # back. Test it explicitly with --ep if you ever want it.
    skip = {'CPUExecutionProvider', 'TensorrtExecutionProvider'}
    eps = args.ep or [p for p in available if p not in skip] + ['CPUExecutionProvider']
    print(f'onnxruntime {ort.__version__}, compiled with: {", ".join(available)}')
    print(f'testing: {", ".join(eps)}\n')

    for ep in eps:
        if ep not in available:
            print(f'--- {ep}: not in this build, skipping\n')
            continue
        print(f'--- {ep} ---')
        try:
            s0 = session(args.src, ep)
            feeds = make_input(s0, args.batch)

            a = s0.run(None, feeds)
            b = session(fused_path, ep).run(None, feeds)
            worst = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
            verdict = 'BIT-EXACT' if worst == 0.0 else f'DIFFERS ({worst:.3e}) -- INVESTIGATE'
            print(f'  parity        {verdict}')

            for label, path in (('as exported', args.src), ('fused', fused_path)):
                counts, cpu_ops = placement(path, ep, feeds)
                total = sum(counts.values())
                brief = '  '.join(f'{k.replace("ExecutionProvider", "")}={v}'
                                  for k, v in counts.most_common())
                print(f'  placement {label:12s} {total:4d} node-runs   {brief}')
                if ep != 'CPUExecutionProvider' and cpu_ops:
                    top = '  '.join(f'{k}={v}' for k, v in cpu_ops.most_common(5))
                    print(f'{"":28s}on CPU: {top}')

            t0 = timeit(args.src, ep, feeds, args.repeats)
            t1 = timeit(fused_path, ep, feeds, args.repeats)
            print(f'  as exported   {t0:8.1f} ms')
            print(f'  fused         {t1:8.1f} ms   {t0 / t1:.2f}x'
                  f'{"   <-- REGRESSION" if t1 > t0 * 1.02 else ""}')
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {str(e)[:200]}')
        print()


if __name__ == '__main__':
    main()
