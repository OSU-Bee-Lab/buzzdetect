import os, sys, time, threading, numpy as np, soxr, soundfile as sf
sys.path.insert(0, '.')
from src.stream.drivers.mp3 import LocalDriver, Driver

P = "/media/server storage/experiments/Chia - Solar Eclipse/11/240408_1249.mp3"
CHUNK = 500

def best(fn, n=3):
    return min((lambda t=time.perf_counter(): (fn(), time.perf_counter()-t)[1])() for _ in range(n))

def main():
    sr = sf.info(P).samplerate
    n = int(CHUNK * sr)

    d = LocalDriver(P); d.seek(0)
    t_shim = best(lambda: d.read(n)); d.close()

    s = sf.SoundFile(P); s.seek(0)
    t_plain = best(lambda: s.read(n, dtype='float32')); s.close()

    os.environ['BUZZDETECT_MP3_HELPERS'] = 'always'
    h = Driver(P)
    used_helper = h._helper is not None
    h.seek(0); t_help = best(lambda: h.read(n)); h.close()

    raw = np.zeros(n, dtype=np.float32)
    t_res = best(lambda: soxr.resample(raw, sr, 16000, quality='HQ'))

    print(f"per {CHUNK}s chunk @ {sr}Hz ({n*4/2**20:.0f} MB float32)\n")
    for label, t in [('plain soundfile read', t_plain),
                     ('mp3 driver, in-process (shim)', t_shim),
                     (f'mp3 driver, helper process ({"real" if used_helper else "FELL BACK"})', t_help)]:
        print(f'  {label:<44} {t*1000:7.1f} ms   {CHUNK/t:8.0f}x')
    print(f'\n  resample 44.1k -> 16k (soxr HQ)              {t_res*1000:7.1f} ms')
    print(f'\n  streamer total, plain sf  : {(t_plain+t_res)*1000:7.1f} ms -> {CHUNK/(t_plain+t_res):6.0f}x/thread'
          f'  => 8 threads ideal {8*CHUNK/(t_plain+t_res):7.0f}x')
    print(f'  streamer total, shim      : {(t_shim+t_res)*1000:7.1f} ms -> {CHUNK/(t_shim+t_res):6.0f}x/thread'
          f'  => 8 threads ideal {8*CHUNK/(t_shim+t_res):7.0f}x')
    print(f'  streamer total, helper    : {(t_help+t_res)*1000:7.1f} ms -> {CHUNK/(t_help+t_res):6.0f}x/thread'
          f'  => 8 threads ideal {8*CHUNK/(t_help+t_res):7.0f}x')

if __name__ == '__main__':
    import multiprocessing; multiprocessing.set_start_method('spawn', force=True)
    main()
