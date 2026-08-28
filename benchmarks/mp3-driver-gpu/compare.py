"""Both metrics for a log: the R script's steady-state rate, and total audio / total wall."""
import re, sys, glob
from datetime import datetime
CHUNK = re.compile(r'chunk \((.*?)\) in')
TS = re.compile(r'^(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)')

def parse(path):
    chunks, stamps = [], []
    for line in open(path, errors='replace'):
        m = TS.match(line.replace(',', '.'))
        if m:
            stamps.append(datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'))
        if 'rate' not in line:
            continue
        c = CHUNK.search(line)
        if not c or not m:
            continue
        a, b = c.group(1).split(', ')
        chunks.append((stamps[-1], float(b) - float(a)))
    return chunks, stamps

def report(path, wall_override=None):
    chunks, stamps = parse(path)
    if len(chunks) < 3:
        return None
    audio = sum(c[1] for c in chunks)
    log_span = (max(stamps) - min(stamps)).total_seconds()
    tail = chunks[1:]
    r_span = (max(t for t, _ in tail) - min(t for t, _ in tail)).total_seconds()
    r_rate = sum(c[1] for c in tail) / r_span if r_span > 0 else 0
    wall = wall_override or log_span
    return dict(path=path, n=len(chunks), audio_h=audio/3600,
                r_rate=r_rate, wall=wall, total_rate=audio/wall if wall else 0)

if __name__ == '__main__':
    override = float(sys.argv[1]) if sys.argv[1] != '-' else None
    print(f"{'R-metric':>10} {'audio/wall':>11} {'audio h':>8} {'wall s':>8}  log")
    for pat in sys.argv[2:]:
        for p in sorted(glob.glob(pat, recursive=True)):
            d = report(p, override if len(sys.argv[2:]) == 1 else None)
            if d:
                print(f"{d['r_rate']:9.0f}x {d['total_rate']:10.0f}x {d['audio_h']:8.1f} {d['wall']:8.0f}  {p}")
