"""Run one analysis with the mp3 read path selected by argv[1].

Arms:
  soundfile          plain soundfile, as before drivers/mp3.py existed. Loses
                     the last 0.17% of every long file, so it analyses slightly
                     less audio; the rate is computed from audio actually
                     analysed and stays comparable.
  scan, scan-local   the driver as it was before the tail-scan path: mpg123's
                     whole-file length scan, and every read of the file's life
                     through the shim. `scan` leaves the helper policy alone,
                     `scan-local` forces in-process.
  default            the tail-scan path as shipped, helper policy left alone.
  force-helper       the same, with every open pushed into a helper process.
  nohelper           the same, with helpers refused outright.

The __main__ guard is load-bearing: the mp3 driver's helpers use a spawn
context, and spawn re-imports this file in every child. Without it, each helper
re-runs the whole analysis -- several concurrent analyses fighting over one GPU,
which looks exactly like a performance regression.
"""
import os
import sys


def main():
    sys.path.insert(0, '.')
    arm, dir_out = sys.argv[1], sys.argv[2]

    if arm == 'soundfile':
        import soundfile as sf
        from src.stream import audio
        audio.driver_map['mp3'] = sf.SoundFile
    elif arm in ('scan', 'scan-local'):
        # An environment variable rather than a patched module, because the
        # helper processes are spawned and re-import everything: a patch applied
        # here would not reach them, and the arm would quietly measure the new
        # read path in the children.
        os.environ['BUZZDETECT_MP3_TAILSCAN'] = 'never'
        if arm == 'scan-local':
            os.environ['BUZZDETECT_MP3_HELPERS'] = 'never'
    elif arm == 'nohelper':
        os.environ['BUZZDETECT_MP3_HELPERS'] = 'never'
    elif arm == 'force-helper':
        os.environ['BUZZDETECT_MP3_HELPERS'] = 'always'
    elif arm != 'default':
        sys.exit(f'unknown arm {arm}')

    from src.analyze import analyze
    analyze(modelname='model_general_v3',
            dir_audio='/media/server storage/experiments/Chia - Solar Eclipse',
            dir_out=dir_out, chunklength=500, n_streamers=8,
            stream_buffer_depth=8, analyzers_gpu=1, analyzers_cpu=0,
            verbosity_print='PROGRESS', log_progress=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
