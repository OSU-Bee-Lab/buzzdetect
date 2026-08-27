#!/usr/bin/env python3
"""Sweep streamer count against chunk length and report the analysis rate.

Rate is audio seconds over wall seconds for the whole run -- the number that
decides how long a real analysis takes -- taken from the engine's own
`file_start` progress events (exact durations) over the wall clock of the
subprocess. Everything else in the CSV is there to explain a rate.

Re-runnable: results append to the CSV and completed cells are skipped, so an
interrupted sweep resumes where it stopped. `--force` re-runs everything.

    python3 run_grid.py --out results.csv                    # full grid
    python3 run_grid.py --dry-run                            # show the corpus
    python3 run_grid.py --streamers 6 --chunklengths 200     # one cell

GPU runs need an interpreter whose onnxruntime carries a GPU execution
provider; point --python at it. On a machine with more than one cuDNN on the
loader path, pin --ld-library-path at the one that works, or the run dies on
its first FusedConv (see engine/src/inference/onnx.py).
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
ENGINE = os.path.join(REPO, 'engine')

# Every corpus recording is 48 kbps CBR, so bytes/6000 is audio seconds to
# better than 0.1%. Only used to choose files against a duration budget; the
# durations that reach the CSV come from the engine.
BYTES_PER_AUDIO_SECOND = 6000

FIELDS = [
    'timestamp', 'streamers', 'chunklength', 'status',
    'rate_x', 'audio_s', 'wall_s',
    'engine_s', 'peak_rss_mb', 'files_started', 'files_expected',
    'processor', 'analyzers', 'model', 'corpus_files', 'corpus_hours_est',
    'note', 'log',
]

# Allocator wording that means "ran out of memory" rather than a real fault.
# The engine raises a MemoryError naming chunklength for these, but a run can
# also die before that -- or be killed outright by the kernel.
OOM_PATTERNS = re.compile(
    r'ran out of memory|MemoryError|out of memory|cudaErrorMemoryAllocation|'
    r'failed to allocate memory|CUBLAS_STATUS_ALLOC_FAILED',
    re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus-root', default='/media/server storage/experiments/Luke - Diel Drivers',
                   help='Directory of dated corpus folders to draw files from.')
    p.add_argument('--corpus-dirs', nargs='*',
                   default=['2026-08-21', '2026-08-18', '2026-08-15', '2026-08-04', '2026-07-22'],
                   help='Which dated folders to draw from, in order.')
    p.add_argument('--target-hours', type=float, default=400.0,
                   help='Audio-hour budget for the corpus. Files are taken one '
                        'per subfolder, round-robin, until the budget is met.')
    p.add_argument('--min-files', type=int, default=None,
                   help='Take at least this many files whatever the budget says. '
                        'Defaults to the largest streamer count, so every cell '
                        'has at least one file per streamer. Below that a run '
                        'measures idle streamers rather than contention.')
    p.add_argument('--streamers', type=int, nargs='+', default=[1, 6, 12])
    p.add_argument('--chunklengths', type=float, nargs='+', default=[100, 200, 600, 1200])
    p.add_argument('--processor', choices=['gpu', 'cpu'], default='gpu')
    p.add_argument('--analyzers', type=int, default=1)
    p.add_argument('--model', default='model_general_v3')
    p.add_argument('--repeat', type=int, default=1, help='Runs per cell.')
    p.add_argument('--timeout', type=float, default=7200, help='Seconds per run.')
    p.add_argument('--python', default=os.environ.get(
        'BUZZDETECT_PYTHON', os.path.join(ENGINE, '.venv', 'bin', 'python3')))
    p.add_argument('--ld-library-path', default=os.environ.get('BUZZDETECT_LD_LIBRARY_PATH'),
                   help='LD_LIBRARY_PATH for the engine subprocess.')
    p.add_argument('--work-dir', default=os.path.join(HERE, 'work'),
                   help='Staging (symlinks), per-cell outputs and logs.')
    p.add_argument('--out', default=os.path.join(HERE, 'results.csv'))
    p.add_argument('--no-warmup', action='store_true',
                   help='Skip the read-through that warms the page cache. The '
                        'warmup exists so the first cell is not penalised for '
                        'reading cold what every later cell reads from cache.')
    p.add_argument('--force', action='store_true', help='Re-run completed cells.')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def select_corpus(args):
    """Files to analyse: one per subfolder, round-robin, to a duration budget.

    Round-robin rather than folder-by-folder so a short corpus still spans
    recorders and days, and so consecutive files are not neighbours on disk.
    """
    by_folder = []
    for name in args.corpus_dirs:
        root = os.path.join(args.corpus_root, name)
        if not os.path.isdir(root):
            print(f'warning: no such corpus dir, skipping: {root}', file=sys.stderr)
            continue
        for folder in sorted(os.scandir(root), key=lambda e: e.name):
            if not folder.is_dir():
                continue
            files = sorted(
                os.path.join(folder.path, f)
                for f in os.listdir(folder.path) if f.lower().endswith('.mp3'))
            if files:
                by_folder.append(files)

    min_files = args.min_files if args.min_files is not None else max(args.streamers)
    budget = args.target_hours * 3600
    chosen, total, depth = [], 0.0, 0
    while any(len(f) > depth for f in by_folder):
        for files in by_folder:
            if depth >= len(files):
                continue
            if total >= budget and len(chosen) >= min_files:
                return chosen, total
            path = files[depth]
            try:
                total += os.stat(path).st_size / BYTES_PER_AUDIO_SECOND
            except OSError:
                continue
            chosen.append(path)
        depth += 1
    return chosen, total


def stage(paths, work_dir):
    """A flat directory of symlinks, so results land in one place per cell."""
    staged = os.path.join(work_dir, 'audio')
    os.makedirs(staged, exist_ok=True)
    for name in os.listdir(staged):
        os.unlink(os.path.join(staged, name))
    for i, path in enumerate(paths):
        # Numbered so names stay unique across folders that reuse them.
        link = os.path.join(staged, f'{i:03d}_{os.path.basename(path)}')
        os.symlink(path, link)
    return staged


def warm_cache(paths):
    t0 = time.perf_counter()
    total = 0
    for path in paths:
        try:
            with open(path, 'rb') as f:
                while f.read(1 << 24):
                    total += 1 << 24
        except OSError:
            pass
    print(f'  warmed {total / 2**30:.1f} GiB in {time.perf_counter() - t0:.0f}s',
          flush=True)


def audio_seconds(text):
    """Total audio the engine reported starting, from its progress events."""
    total, started = 0.0, 0
    for line in text.splitlines():
        if not line.startswith('BDPROGRESS '):
            continue
        try:
            event = json.loads(line[len('BDPROGRESS '):])
        except ValueError:
            continue
        if event.get('event') == 'file_start':
            total += float(event.get('duration') or 0)
            started += 1
    return total, started


def classify(returncode, text, timed_out):
    if timed_out:
        return 'timeout', 'exceeded --timeout'
    if returncode == 0 and 'All files analyzed and cleaned.' in text:
        return 'ok', ''
    if OOM_PATTERNS.search(text):
        return 'oom', 'ran out of memory'
    # SIGKILL with no traceback is almost always the kernel's OOM killer,
    # which gives the process no chance to report anything. It reaches us two
    # ways: as a negative returncode, or -- because /usr/bin/time wraps the
    # engine -- as time's own report of how its child died.
    if returncode == -9 or 'Command terminated by signal 9' in text:
        return 'oom', 'killed (SIGKILL), likely kernel OOM killer'
    reason = ''
    for line in text.splitlines():
        if 'cannot continue' in line:
            reason = line.strip()[:300]
            break
    return 'failed', reason or f'exit {returncode}'


def run_cell(args, staged, streamers, chunklength, index):
    tag = f's{streamers}_c{int(chunklength)}_{index}'
    out_dir = os.path.join(args.work_dir, 'out', tag)
    log_path = os.path.join(args.work_dir, 'logs', f'{tag}.log')
    for d in (out_dir, os.path.dirname(log_path)):
        os.makedirs(d, exist_ok=True)
    # A fresh output directory per cell: the engine skips files it has already
    # analysed, so reusing one would measure nothing.
    for name in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, name))

    gpu = args.processor == 'gpu'
    cmd = ['/usr/bin/time', '-f', 'BENCHRUSAGE %M %e',
           args.python, 'buzzdetect_cli.py',
           '--modelname', args.model,
           '--dir_audio', staged,
           '--dir_out', out_dir,
           '--chunklength', str(chunklength),
           '--n_streamers', str(streamers),
           '--analyzers_gpu', str(args.analyzers if gpu else 0),
           '--analyzers_cpu', str(0 if gpu else args.analyzers),
           '--verbosity_print', 'PROGRESS',
           '--log_progress', 'false']

    env = dict(os.environ)
    if args.ld_library_path:
        env['LD_LIBRARY_PATH'] = args.ld_library_path

    t0 = time.perf_counter()
    timed_out = False
    # Own process group: on timeout the engine's streamers, writer and mp3
    # helper processes all have to go, and killing the direct child (which is
    # /usr/bin/time) would leave them running and skew every later cell.
    proc = subprocess.Popen(cmd, cwd=ENGINE, env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        text = proc.communicate(timeout=args.timeout)[0].decode('utf-8', 'replace')
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out, returncode = True, None
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()
        text = (proc.communicate()[0] or b'').decode('utf-8', 'replace')
    wall = time.perf_counter() - t0

    with open(log_path, 'w') as f:
        f.write(text)

    peak_rss_mb = ''
    rusage = re.search(r'BENCHRUSAGE (\d+) ([\d.]+)', text)
    if rusage:
        peak_rss_mb = round(int(rusage.group(1)) / 1024, 1)

    status, note = classify(returncode, text, timed_out)
    audio_s, started = audio_seconds(text)
    engine_s = ''
    m = re.search(r'Total analysis time: ([\d,.]+)s', text)
    if m:
        engine_s = m.group(1).replace(',', '')

    # A partial run's rate would flatter or damn it arbitrarily depending on
    # how far it got, so only completed runs get one.
    rate = round(audio_s / wall, 1) if status == 'ok' and wall > 0 else ''

    return {
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'streamers': streamers, 'chunklength': chunklength, 'status': status,
        'rate_x': rate, 'audio_s': round(audio_s, 1), 'wall_s': round(wall, 1),
        'engine_s': engine_s, 'peak_rss_mb': peak_rss_mb,
        'files_started': started, 'processor': args.processor,
        'analyzers': args.analyzers, 'model': args.model,
        'note': note, 'log': os.path.relpath(log_path, HERE),
    }


def load_done(path):
    if not os.path.exists(path):
        return set(), []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    done = {(r['streamers'], r['chunklength']) for r in rows if r['status'] == 'ok'}
    return done, rows


def main():
    args = parse_args()
    paths, est_seconds = select_corpus(args)
    if not paths:
        sys.exit('no audio found; check --corpus-root and --corpus-dirs')

    print(f'corpus: {len(paths)} files, ~{est_seconds / 3600:.0f} audio hours '
          f'from {", ".join(args.corpus_dirs)}')
    if args.dry_run:
        for p in paths:
            print(f'  {os.stat(p).st_size / BYTES_PER_AUDIO_SECOND / 3600:6.1f}h  {p}')
        return

    os.makedirs(args.work_dir, exist_ok=True)
    staged = stage(paths, args.work_dir)

    if not args.no_warmup:
        print('warming page cache...', flush=True)
        warm_cache(paths)

    done, _ = ((set(), []) if args.force else load_done(args.out))
    new_file = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if new_file:
            writer.writeheader()

        cells = [(s, c) for c in args.chunklengths for s in args.streamers]
        for n, (streamers, chunklength) in enumerate(cells, 1):
            key = (str(streamers), str(chunklength))
            if key in done:
                print(f'[{n}/{len(cells)}] streamers={streamers} '
                      f'chunk={chunklength}s -- already done, skipping', flush=True)
                continue
            for i in range(args.repeat):
                print(f'[{n}/{len(cells)}] streamers={streamers} '
                      f'chunk={chunklength}s run {i + 1}/{args.repeat}...',
                      end=' ', flush=True)
                row = run_cell(args, staged, streamers, chunklength, i)
                row['corpus_files'] = len(paths)
                row['corpus_hours_est'] = round(est_seconds / 3600, 1)
                row['files_expected'] = len(paths)
                writer.writerow(row)
                f.flush()
                if row['status'] == 'ok':
                    print(f"{row['rate_x']}x  ({row['wall_s']}s wall, "
                          f"{row['peak_rss_mb']}MB peak)", flush=True)
                else:
                    print(f"{row['status'].upper()} -- {row['note']}", flush=True)

    print(f'\nresults: {args.out}')


if __name__ == '__main__':
    main()
