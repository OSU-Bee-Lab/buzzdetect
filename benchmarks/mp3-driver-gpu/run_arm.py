"""Run one analysis with the mp3 read path selected by argv[1].

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
        # What the 2025-09 runs did, because drivers/mp3.py did not exist yet.
        # Truncates long files, so this arm analyses slightly less audio; rate
        # is computed from audio actually analysed, so it stays comparable.
        import soundfile as sf
        from src.stream import audio
        audio.driver_map['mp3'] = sf.SoundFile
    elif arm == 'nohelper':
        os.environ['BUZZDETECT_MP3_HELPERS'] = 'never'
    elif arm != 'helper':
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
