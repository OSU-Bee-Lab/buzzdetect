"""Raw ONNX inference speed for every model in engine/models/.

Times session.run() on one chunk of audio and nothing else: no streaming,
decoding, resampling or result writing. Each model is timed at fp32 and, where
model.fp16.onnx exists, fp16. Audio is synthetic noise -- the graphs have no
data-dependent control flow, so content doesn't change the speed.

Run from anywhere; it chdirs to engine/ (src/config.py uses relative paths).

    engine/.venv/bin/python3 benchmarks/model-speed/bench_models.py
    engine/.venv/bin/python3 benchmarks/model-speed/bench_models.py --processor CPU   # or GPU; default is both
    engine/.venv/bin/python3 benchmarks/model-speed/bench_models.py --seconds 200 --repeats 15

Default runs CPU and GPU, each at fp32 and (where a model has one) fp16, and prints one
table. A GPU column is omitted if no GPU provider actually loads.

On CPU the fp16 graph is loaded directly; on GPU it goes through
BUZZDETECT_GPU_FP16=1, exactly as the engine does.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'engine')
os.chdir(ENGINE)
sys.path.insert(0, os.getcwd())

from src.inference import onnx as bd_onnx  # noqa: E402


def discover(root='models'):
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, 'model.onnx')) and os.path.isfile(os.path.join(d, 'config_model.json')):
            yield name, d


def session_samples(config, seconds):
    """Same arithmetic as OnnxModel.session_length / n_frames."""
    n = int(round(seconds * config['samplerate']))
    hop = np.float32(1.0) / np.float32(config['samples_hop'])
    frames = 1 + int(np.ceil(np.float32(max(0, n - config['samples_min'])) * hop))
    return n, config['samples_min'] + config['samples_hop'] * frames


def bench(path, processor, fp16, samples_session, audio, repeats, warmup):
    if fp16:
        os.environ[bd_onnx.ENV_ALLOW_FP16] = '1'
    else:
        os.environ.pop(bd_onnx.ENV_ALLOW_FP16, None)

    t0 = time.perf_counter()
    session = bd_onnx.make_session(path, processor, samples_session)
    t_load = time.perf_counter() - t0
    name_in = session.get_inputs()[0].name

    padded = np.zeros(samples_session, dtype=np.float32)
    padded[:len(audio)] = audio
    feed = {name_in: padded}

    for _ in range(warmup):
        session.run(None, feed)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t)
    return t_load, times, session.get_providers()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=200)
    ap.add_argument('--repeats', type=int, default=15)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--processor', choices=['CPU', 'GPU', 'both'], default='both')
    ap.add_argument('--models', nargs='*', help='names to run (default: all)')
    args = ap.parse_args()
    processors = ['CPU', 'GPU'] if args.processor == 'both' else [args.processor]

    rng = np.random.default_rng(0)
    results = {}   # (model, processor, precision) -> (mean, se) or None if unavailable
    providers = {}  # processor -> provider that actually ran
    models = []
    for name, d in discover():
        if args.models and name not in args.models:
            continue
        models.append(name)
        with open(os.path.join(d, 'config_model.json')) as f:
            config = json.load(f)
        n, samples_session = session_samples(config, args.seconds)
        audio = rng.uniform(-0.1, 0.1, n).astype(np.float32)

        variants = [('fp32', 'model.onnx')]
        if os.path.isfile(os.path.join(d, bd_onnx.FNAME_FP16)):
            variants.append(('fp16', bd_onnx.FNAME_FP16))

        for processor in processors:
            for label, fname in variants:
                fp16 = label == 'fp16'
                path = os.path.abspath(os.path.join(d, fname))
                # On CPU there is no fp16 switch, so hand over the fp16 file itself.
                # On GPU, make_session takes the sibling from the env var.
                if processor == 'GPU':
                    path = os.path.abspath(os.path.join(d, 'model.onnx'))
                try:
                    t_load, times, provider = bench(path, processor, fp16, samples_session,
                                                    audio, args.repeats, args.warmup)
                except Exception as e:
                    print(f'{name} {processor} {label}: FAILED {type(e).__name__}: {e}', file=sys.stderr)
                    continue
                if processor == 'GPU' and provider == 'CPUExecutionProvider':
                    print(f'{name} {processor} {label}: no GPU provider loaded, skipped', file=sys.stderr)
                    continue
                rates = [args.seconds / t for t in times]
                mean = statistics.mean(rates)
                se = statistics.stdev(rates) / len(rates) ** 0.5
                results[(name, processor, label)] = (mean, se)
                providers[processor] = provider
                print(f'{name:28s} {processor} {label} {mean:8.0f} +- {se:.0f} s/s', flush=True)

    print(f'\n{args.seconds:g} s of audio per run, {args.repeats} runs ({args.warmup} warmup)')
    print('rate = audio seconds analyzed per wall-clock second, mean +- standard error')
    for proc, prov in providers.items():
        print(f'{proc} = {prov}')
    print()

    cols = [(p, l) for p in processors for l in ('fp32', 'fp16')
            if any((m, p, l) in results for m in models)]
    print(f'{"model":28s}' + ''.join(f'{p + " " + l:>16s}' for p, l in cols))
    for m in models:
        cells = []
        for p, l in cols:
            r = results.get((m, p, l))
            cells.append(f'{r[0]:9.0f} +-{r[1]:3.0f}' if r else f'{"-":>14s}')
        print(f'{m:28s}' + ''.join(f'{c:>16s}' for c in cells))


if __name__ == '__main__':
    main()
