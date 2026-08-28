"""Is libsndfile's mp3 length a wrong *report*, or a hard *clamp* on reads?"""
import sys, time, numpy as np, soundfile as sf
sys.path.insert(0, '.')
from src.stream.drivers.mp3 import LocalDriver

P = sys.argv[1]

d = LocalDriver(P); true_frames = d.frames; sr = d.samplerate; d.close()
s = sf.SoundFile(P); est_frames = s.frames

print(f'{P.split("/")[-1]}  @{sr}Hz')
print(f'  libsndfile estimate : {est_frames:>12,} frames  ({est_frames/sr/3600:6.3f} h)')
print(f'  true (scanned)      : {true_frames:>12,} frames  ({true_frames/sr/3600:6.3f} h)')
missing = true_frames - est_frames
print(f'  missing             : {missing:>12,} frames  ({missing/sr:.1f} s, {100*missing/true_frames:.3f}%)\n')

# Sit just before the estimated end and try to read well past it.
s.seek(max(0, est_frames - sr))
got = s.read(int(sr * 600), dtype='float32')
print(f'  seek(est-1s) then read 600s -> got {len(got):,} frames '
      f'({len(got)/sr:.1f}s); {"CLAMPED at estimate" if len(got) <= sr*1.05 else "READ PAST ESTIMATE"}')

# And can we seek past the estimate at all?
try:
    pos = s.seek(est_frames + sr * 10)
    after = s.read(sr * 5, dtype='float32')
    print(f'  seek(est+10s) -> pos {pos:,}; read got {len(after):,} frames')
except Exception as e:
    print(f'  seek(est+10s) -> raised {type(e).__name__}: {e}')
s.close()
