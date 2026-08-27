"""Targeted inference benchmark: ONNX vs TensorFlow, same audio, same batch shape.

Run once per runtime, in its own process (the two runtimes would otherwise
contend for a 4GB card). Must be run with engine/ as the working directory --
src/config.py addresses models/ and embedders/ by relative path.

    BUZZDETECT_RUNTIME=onnx python bench_inference.py --out results_onnx.json

No engine code is modified; this only calls into it.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

# This script lives outside the repo, so sys.path[0] is its own directory. The
# engine addresses itself as `src.*` / `embedders.*` relative to the working
# directory (which must be engine/ anyway, since src/config.py uses relative
# paths). tools/onnxify_model.py does the same thing.
sys.path.insert(0, os.getcwd())

CHUNK_SECONDS = 200.0          # matches --chunklength 200 in the end-to-end runs
SAMPLERATE = 16000


def load_chunk(path_audio, seconds, offset_s=0.0):
    """Decode and resample one chunk exactly the way stream/worker.py does."""
    import soxr
    from src.stream.audio import build_track

    track = build_track(path_audio)
    try:
        sample_from = int(offset_s * track.samplerate)
        read_size = int(seconds * track.samplerate)
        track.seek(sample_from)
        samples = track.read(read_size, dtype=np.float32)
        if track.channels > 1:
            samples = np.mean(samples, axis=1)
        if len(samples) < read_size:
            raise RuntimeError(
                f'wanted {read_size} samples from {path_audio}, got {len(samples)}')
        samples = soxr.resample(samples, track.samplerate, SAMPLERATE, quality='HQ')
        return samples.astype(np.float32)
    finally:
        track.close()


def time_it(fn, repeats, warmup, force):
    """Median-of-repeats wall time for fn, after `warmup` untimed calls.

    `force` is applied to the return value inside the timed region. For
    TensorFlow that is the sync point: eager ops return once the work is
    *enqueued* on the CUDA stream, so timing without materialising the result
    measures dispatch, not compute. onnxruntime's run() is already synchronous,
    so force is a no-op there and costs only a cheap np.asarray.
    """
    for _ in range(warmup):
        force(fn())

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        force(out)
        times.append(time.perf_counter() - t0)
    return times


def summarize(times, audio_seconds):
    med = statistics.median(times)
    return {
        'median_s': med,
        'min_s': min(times),
        'mean_s': statistics.mean(times),
        'stdev_s': statistics.stdev(times) if len(times) > 1 else 0.0,
        'rate_median': audio_seconds / med,   # audio seconds per wall second
        'n': len(times),
        'all_s': times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--modelname', default='model_general_v3')
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--seconds', type=float, default=CHUNK_SECONDS)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    report = {
        'runtime_requested': os.environ.get('BUZZDETECT_RUNTIME', 'auto'),
        'python': sys.version.split()[0],
        'chunk_seconds': args.seconds,
        'repeats': args.repeats,
        'warmup': args.warmup,
    }

    # Decode before touching either runtime, so the audio is identical and no
    # decode cost lands inside a timed region.
    samples = load_chunk(args.audio, args.seconds)
    report['n_samples'] = int(len(samples))
    report['audio_seconds_actual'] = len(samples) / SAMPLERATE

    from src.inference.models import load_model, DualRuntimeModel

    model = load_model(args.modelname, framehop_prop=1, initialize=False)
    model.processor = 'GPU'

    # Three arms, and only two of them are DualRuntimeModels. The third,
    # model_general_v3_onnx, is a plain BaseModel wrapping model_combined.onnx:
    # one graph taking a raw waveform, with the log-mel front end fused in
    # rather than run in NumPy on the CPU. It has no .runtime, and it inherits
    # BaseModel's uses_tensorflow=True default despite being pure onnxruntime,
    # so neither can be trusted here.
    is_dual = isinstance(model, DualRuntimeModel)
    is_fused_onnx = not is_dual and 'onnx' in args.modelname
    if is_dual:
        arm = model.runtime
        uses_tf = model.uses_tensorflow
    elif is_fused_onnx:
        arm = 'onnx_fused'
        uses_tf = False
    else:
        arm = 'unknown'
        uses_tf = model.uses_tensorflow

    report['arm'] = arm
    report['runtime_chosen'] = arm
    report['embedder'] = model.embeddername
    report['is_fused_onnx'] = is_fused_onnx

    # The fused model declares embeddername 'yamnet_k2', so loading it imports
    # tensorflow even though it never runs a TF op (the embedder is deliberately
    # left uninitialized). Cap its memory anyway -- an idle TF that grabs the
    # card would poison the measurement on a 4GB GTX 1650.
    if uses_tf or is_fused_onnx:
        # Mirror WorkerInferer._managememory: without this TF takes the whole
        # card, which on a 4GB GTX 1650 with a display attached is a problem.
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        report['tf_version'] = tf.__version__
        report['tf_gpus'] = [d.name for d in tf.config.list_physical_devices('GPU')]
    if not uses_tf:
        import onnxruntime as ort
        report['ort_version'] = ort.__version__

    t_init = time.perf_counter()
    model.initialize()
    report['initialize_s'] = time.perf_counter() - t_init

    if is_fused_onnx:
        report['ort_providers_fused'] = model._session.get_providers()
    elif not uses_tf:
        report['ort_providers_embedder'] = model.embedder.model.get_providers()
        report['ort_providers_head'] = model.model.get_providers()

    audio_s = report['audio_seconds_actual']
    sync = np.asarray                     # forces TF to materialise
    noop = lambda x: x

    results = {}

    # Whole prediction: featurisation + trunk + head, as process_chunk calls it.
    results['predict_synced'] = summarize(
        time_it(lambda: model.predict(samples), args.repeats, args.warmup, sync), audio_s)

    # Same call, without materialising the result -- what the analyzer thread
    # actually times today, since np.asarray happens later in write/worker.py.
    # For ONNX this should match predict_synced; any gap is the TF-only bias.
    results['predict_unsynced'] = summarize(
        time_it(lambda: model.predict(samples), args.repeats, args.warmup, noop), audio_s)

    # Trunk and head separately -- not possible for the fused arm, where the
    # whole point is that front end, trunk and head are one graph.
    if not is_fused_onnx:
        # Trunk alone (YAMNet). Nearly all of the compute.
        results['embed_synced'] = summarize(
            time_it(lambda: model.embedder.embed(samples), args.repeats, args.warmup, sync),
            audio_s)

        # Head alone (one dense layer), on a fixed embedding.
        embeddings = np.asarray(model.embedder.embed(samples))
        report['embedding_shape'] = list(embeddings.shape)
        results['head_synced'] = summarize(
            time_it(lambda: model.predict_embeddings(embeddings), args.repeats, args.warmup, sync),
            audio_s)

    # The ONNX trunk splits in a way the TF one does not: its log-mel front end
    # runs in NumPy on the CPU (embedders/yamnet_onnx/embedder.py), while the
    # TensorFlow embedder computes the same front end inside the Keras graph on
    # the GPU. Splitting it here shows whether featurisation alone accounts for
    # any gap.
    if arm == 'onnx':
        from embedders.yamnet_onnx import features
        params = model.embedder.params
        results['onnx_featurize_cpu'] = summarize(
            time_it(lambda: features.waveform_to_patches(samples, params),
                    args.repeats, args.warmup, noop), audio_s)

        patches = features.waveform_to_patches(samples, params)
        report['patches_shape'] = list(patches.shape)
        name_in = model.embedder.name_in
        results['onnx_trunk_only'] = summarize(
            time_it(lambda: model.embedder.model.run(None, {name_in: patches}),
                    args.repeats, args.warmup, noop), audio_s)

    report['results'] = results

    # Full-precision output, for the parity check. It cannot be done from the
    # result CSVs: write/formatting.py:33 rounds to digits_results (2 for this
    # model), which cannot resolve the 5.6e-05 agreement claimed at export.
    path_npy = os.path.splitext(args.out)[0] + '_predictions.npy'
    np.save(path_npy, np.asarray(model.predict(samples), dtype=np.float64))
    report['predictions_npy'] = path_npy


    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"runtime: {report['runtime_chosen']}  embedder: {report['embedder']}")
    print(f"initialize: {report['initialize_s']:.2f}s")
    for name, r in results.items():
        print(f"  {name:22s} median {r['median_s']*1000:8.1f} ms   "
              f"rate {r['rate_median']:8.1f} audio-s/wall-s")


if __name__ == '__main__':
    main()
