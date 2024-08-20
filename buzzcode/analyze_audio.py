import warnings

from buzzcode.utils import search_dir, Timer, setthreads

setthreads(1)

from buzzcode.embeddings import load_embedder
from buzzcode.analysis import loadup, translate_results, suffix_result, suffix_partial, solve_memory, melt_coverage, \
    get_gaps, smooth_gaps, gaps_to_chunklist, stitch_partial
import pandas as pd
import os
import re
import librosa
import multiprocessing
import json
import tensorflow as tf
import soundfile as sf
import numpy as np
from datetime import datetime

warnings.warn(
    'chunklength calculation is currently done assuming a framehop of 1; need to  update embedder handling so framehop can be read before laoding embedder')

# TODO: have gpu worker sub-chunk if needed for memory!

#  modelname = "model_general"; cpus=4; memory_allot = 3; dir_audio="./audio_in"; dir_out=None; verbosity=1; paths_audio = None; embeddername='yamnet_wholehop'; result_detail='sparse'
def analyze_batch(modelname, cpus, memory_allot, gpu=False, vram=None, embeddername='yamnet_wholehop', dir_audio="./audio_in", verbosity=1,
                  result_detail='rich'):
    timer_total = Timer()

    dir_model = os.path.join("models", modelname)
    dir_out = os.path.join(dir_model, "output")

    log_timestamp = timer_total.time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)
    log = open(path_log, "x")
    log.close()

    control = []  # make control now for worker_logger to reference in case of early exit

    # Logging
    #
    q_log = multiprocessing.Queue()

    def printlog(item, item_verb=0, do_log=True):
        time_current = datetime.now()

        if do_log:
            q_log.put(f"{time_current} - {item} \n")

        if item_verb <= verbosity:
            print(item)

        return item


    def worker_logger():
        workers_running = cpus
        while workers_running > 0:
            log_item = q_log.get(block=True)
            if log_item == 'TERMINATE':
                workers_running -= 1
                continue

            file_log = open(path_log, "a")
            file_log.write(log_item)
            file_log.close()

        # on terminate, clean up chunks
        for c in control:
            base_out = c['path_audio']
            base_out = os.path.splitext(base_out)[0]
            base_out = re.sub(dir_audio, dir_out, base_out)

            stitch_partial(base_out, c['duration_audio'])

        timer_total.stop()
        closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s"

        print(closing_message)
        file_log = open(path_log, "a")
        file_log.write(closing_message)
        file_log.close()

    proc_logger = multiprocessing.Process(target=worker_logger)
    proc_logger.start()

    # File handling
    #

    with open(os.path.join(dir_model, 'config.txt'), 'r') as file:
        config = json.load(file)

    with open(os.path.join(dir_model, 'config.txt'), 'r') as file:
        config_string = file.read()
        framelength_string = re.search(pattern='framelength\":\\ (\\d+\\.\\d+)', string=config_string).group(1)
        framelength_digits = len(framelength_string.split('.')[1])

    framelength = config['framelength']

    concurrent_streamers, buffer_max, chunklength = solve_memory(
        memory_allot=memory_allot,
        cpus=cpus,
        framehop=1  # TODO: fix this so I can read framehop without loading embedder (which kills analysis)
    )

    if chunklength < framelength:
        raise ValueError(f"insufficient memory allotment")

    paths_audio = search_dir(dir_audio, list(sf.available_formats().keys()))

    # start logger early and make these exit prints printlogs?
    if len(paths_audio) == 0:
        printlog(
            f"no compatible audio files found in raw directory {dir_audio} \n"
            f"audio format must be compatible with soundfile module version {sf.__version__} \n"
            "exiting analysis",
            0
        )
        for c in range(cpus):
            q_log.put('TERMINATE')

        return

    for path_audio in paths_audio:
        base_out = re.sub(dir_audio, dir_out, path_audio)
        base_out = os.path.splitext(base_out)[0]

        path_complete = base_out + suffix_result
        if os.path.exists(path_complete):  # if finished analysis file exists, go to next file
            continue

        duration_audio = librosa.get_duration(path=path_audio)
        paths_chunks = search_dir(os.path.dirname(base_out), None)
        paths_chunks = [p for p in paths_chunks if re.search(base_out, p)]

        if not paths_chunks:  # if there are no results, chunklist whole file and move on
            gaps = [(0, duration_audio)]
            chunklist = gaps_to_chunklist(gaps, chunklength)

            control.append(
                {
                    'path_audio': path_audio,
                    'duration_audio': duration_audio,
                    'chunklist': chunklist
                }
            )
            continue

        # if there are results, read them, make chunklist
        # (this is essentially a more efficient implementation of stitch_partial(), so I don't have to stitch, then re-read)
        df = pd.concat([pd.read_csv(p) for p in paths_chunks])
        coverage = melt_coverage(df)

        gaps = get_gaps(
            range_in=(0, duration_audio),
            coverage_in=coverage
        )

        gaps = smooth_gaps(
            gaps,
            range_in=(0, duration_audio),
            framelength=framelength,
            gap_tolerance=framelength / 4
        )

        # if the file is actually finished and it just wasn't stitched, stitch and move on
        if not gaps:
            for p in paths_chunks:
                os.remove(p)
            df.to_csv(path_complete, index=False)
            continue

        chunklist = gaps_to_chunklist(gaps, chunklength)

        control.append(
            {
                'path_audio': path_audio,
                'duration_audio': duration_audio,
                'chunklist': chunklist
            }
        )

        # if the chunks aren't already stitched, stitch 'em
        path_stitched = base_out + suffix_partial
        if paths_chunks != [path_stitched]:
            for p in paths_chunks:
                os.remove(p)
            df.to_csv(path_stitched, index=False)

    if not control:
        printlog(f"all files in {dir_audio} are fully analyzed; exiting analysis",0)
        for c in range(cpus):
            q_log.put('TERMINATE')
        return

    # streamer
    #
    streamers_concurrent = 4  # TODO: tune this. What's best? A single streamer at a time? Two? Ten?
    q_control = multiprocessing.Queue()
    q_assignments = multiprocessing.Queue(maxsize=buffer_max)

    for c in control:
        q_control.put(c)

    for _ in range(streamers_concurrent):
        q_control.put('TERMINATE')

    def worker_streamer(id_streamer):
        c = q_control.get()
        while c != 'TERMINATE':
            printlog(
                f"streamer {id_streamer}: starting on {c['path_audio']}",
                1
            )

            track = sf.SoundFile(c['path_audio'])
            samplerate_native = track.samplerate

            for chunk in c['chunklist']:  # TODO: check for bad audio because samples returned is too small?
                sample_from = int(chunk[0] * samplerate_native)
                sample_to = int(chunk[1] * samplerate_native)
                read_size = sample_to - sample_from

                track.seek(sample_from)
                samples = track.read(read_size)
                if track.channels > 1:
                    samples = np.mean(samples, axis=1)

                # we've found that many of our files give an incorrect .frames count, or else headers are broken
                # this results in a silent failure where no samples are returned
                n_samples = len(samples)
                if n_samples == 0:
                    warnings.warn(
                        f"no data read at time {chunk[0]} for file {c['path_audio']}")
                elif n_samples < read_size:
                    prop = round(n_samples/read_size, 2)

                    warnings.warn(
                        f"unexpectedly small read at time {chunk[0]} for file {c['path_audio']}. "
                        f"Received {prop}% of samples requested ({read_size}/{n_samples})")  # TODO: keep an eye on this; will this occur just because of  normal rounding errors between time and samples?

                samples = librosa.resample(y=samples, orig_sr=samplerate_native, target_sr=config['samplerate'])

                assignment = {
                    'path_audio': c['path_audio'],
                    'chunk': chunk,
                    'duration_audio': duration_audio,
                    'samples': samples
                }

                ######### Temporary debug #########
                if q_assignments.qsize() == buffer_max:
                    print(f"!!! streamer {id_streamer} waiting for free buffer !!!")
                ######### Temporary debug #########

                q_assignments.put(assignment)

    # analyzer
    # 
    def worker_analyzer(id_analyzer, processor):
        if processor == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'

        printlog(f"analyzer {id_analyzer}: processing on {processor}", 1)

        # ready model
        #
        model, config_model = loadup(modelname)
        embedder, config_embedder = load_embedder(embeddername)

        classes = config_model['classes']

        assignment = q_assignments.get()
        timer_analysis = Timer()

        while assignment != 'TERMINATE':
            timer_analysis.restart()

            printlog(f"analyzer {id_analyzer}: analyzing {assignment['path_audio']}, chunk {assignment['chunk']}", 2,do_log=False)

            embeddings = embedder(assignment['samples'])
            results = model(embeddings)
            output = translate_results(np.array(results), classes)

            output.insert(
                column='start',
                value=range(len(output)),
                loc=0
            )

            output['start'] = output['start'] * config_embedder['framelength'] * config_embedder['framehop']
            output['start'] = output['start'] + assignment['chunk'][
                0]  # TODO: there's drift of about -1/600...maybe because of resampling? Floating point error?

            # round to avoid float errors
            output['start'] = round(output['start'], framelength_digits)

            if result_detail.lower() == 'sparse':
                output = output[['start', 'ins_buzz']]

            else:
                output.insert(
                    column='end',
                    value=round(output['start'] + config_embedder['framelength'], framelength_digits),
                    loc=1
                )

                # round to avoid float errors
                output['end'] = round(output['end'], framelength_digits)

            base_out = assignment['path_audio']
            base_out = os.path.splitext(base_out)[0]
            base_out = re.sub(dir_audio, dir_out, base_out)
            path_out = base_out + f"_s{int(assignment['chunk'][0])}" + suffix_partial

            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            output.to_csv(path_out, index=False)

            timer_analysis.stop()
            chunk_duration = assignment['chunk'][1] - assignment['chunk'][0]
            analysis_rate = (chunk_duration / timer_analysis.get_total()).__round__(1)
            printlog(
                f"analyzer {id_analyzer}: analyzed {assignment['path_audio']}, "
                f"chunk {assignment['chunk']} "
                f"in {timer_analysis.get_total()}s (rate: {analysis_rate})",
                2, do_log=False)

            ######### Temporary debug #########
            if q_assignments.qsize() == 0:
                print(f"!!! analyzer {id_analyzer} waiting for assignment !!!")
            ######### Temporary debug #########

            assignment = q_assignments.get()

        printlog(f"analyzer {id_analyzer}: terminating")
        q_log.put('TERMINATE')

    # Go!
    #
    printlog(
        f"begin analysis \n"
        f"start time: {timer_total.time_start} \n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n"
        f"memory allotment {memory_allot}\n",
        0)

    proc_analyzers = []
    for a in range(cpus):
        proc_analyzers.append(
            multiprocessing.Process(target=worker_analyzer, name=f"analysis_proc{a}", args=([a, 'CPU'])))
        proc_analyzers[-1].start()
        pass

    proc_streamers = []
    for s in range(concurrent_streamers):
        proc_streamers.append(multiprocessing.Process(target=worker_streamer, name=f'streamer_proc{s}', args=[s]))
        proc_streamers[-1].start()
        pass


    if gpu:  # things get narsty when running GPU on a multiprocessing.Process(), so just run in main thread last
        worker_analyzer('G', 'GPU')
    # wait for analysis to finish
    proc_logger.join()


if __name__ == "__main__":
    analyze_batch(modelname='model_general', gpu=True, vram=1, cpus=7, memory_allot=10, verbosity=2)
