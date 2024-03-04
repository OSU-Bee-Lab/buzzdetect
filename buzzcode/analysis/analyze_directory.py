from buzzcode.utils import search_dir, Timer, clip_name, setthreads
setthreads(1)

from buzzcode.embeddings import get_embedder
from buzzcode.inputs import extract_input
from buzzcode.analysis.analysis import (solve_memory, get_gaps, get_coverage, gaps_to_chunklist, loadup,
                                        translate_results, merge_chunks)
from buzzcode.audio import load_audio, frame_audio
import os
import re
import sys
import librosa
import multiprocessing
import json
import soundfile as sf
import numpy as np
from datetime import datetime
import warnings

#  modelname = "agricultural_01"; cpus=4; memory_allot = 3; dir_audio="./audio_in"; dir_out=None; verbosity=1; conflict_out="quit"; paths_audio = None
def analyze_batch(modelname, cpus, memory_allot, dir_audio="./audio_in", classes_out = None, paths_audio = None, dir_out=None, verbosity=1):
    timer_total = Timer()

    dir_model = os.path.join("models", modelname)
    with open(os.path.join(dir_model, 'config.txt'), 'r') as file:
        config = json.load(file)

    framelength = config['framelength']

    if dir_out is None:
        dir_out = os.path.join(dir_model, "output")

    chunklength = solve_memory(memory_allot, cpus, framelength=framelength)

    log_timestamp = timer_total.time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)
    log = open(path_log, "x")
    log.close()

    # if result chunks were left behind from previous analyses, clean them
    merge_chunks(dir_out)

    if paths_audio is None:
        paths_audio = search_dir(dir_audio, list(sf.available_formats().keys()))

    # start logger early and make these exit prints printlogs?
    if len(paths_audio) == 0:
        print(
            f"no compatible audio files found in raw directory {dir_audio} \n"
            f"audio format must be compatible with soundfile module version {sf.__version__} \n"
            "exiting analysis"
        )
        sys.exit(0)

    raws_chunklist = []
    raws_unfinished = []

    for path in paths_audio:
        audio_duration = librosa.get_duration(path=path)

        coverage = get_coverage(path, dir_audio, dir_out)
        if len(coverage) == 0:
            coverage = [(0, 0)]

        gaps = get_gaps((0, audio_duration), coverage)

        # ignore gaps that start less than 1 frame from file end
        gaps = [gap for gap in gaps if gap[0] < (audio_duration - framelength)]

        # expand from center gaps smaller than one frame; leave gaps larger than one frame
        gaps = [(gap[0] - framelength/2, gap[0] + framelength/2) if (gap[1] - gap[0]) < framelength else gap for gap in gaps]

        chunklist = gaps_to_chunklist(gaps, chunklength)

        if len(chunklist) > 0:
            raws_unfinished.append(path)
            raws_chunklist.append(chunklist)

    if len(raws_unfinished) == 0:
        print(f"all files in {dir_audio} are fully analyzed; exiting analysis")
        sys.exit(0)

    # process control
    #
    dict_chunk = dict(zip(raws_unfinished, raws_chunklist))

    analyzer_ids = list(range(cpus))
    analyzers_per_raw = (cpus/len(raws_unfinished)).__ceil__()
    # if more analyzers than raws, repeat the list, wrapping the assignment back to the start
    dict_analyzer = {i: (raws_unfinished*analyzers_per_raw)[i] for i in analyzer_ids}

    dict_rawstatus = {p: "finished" for p in paths_audio}
    for p in raws_unfinished:
        dict_rawstatus[p] = "not finished"

    q_request = multiprocessing.Queue()
    q_analyze = [multiprocessing.Queue() for _ in analyzer_ids]
    q_log = multiprocessing.Queue()

    def printlog(item, item_verb=0):
        time_current = datetime.now()
        q_log.put(f"{time_current} - {item} \n")

        if item_verb <= verbosity:
            print(item)

        return item

    # worker definition
    #
    def worker_manager():
        chunks_remaining = sum([len(c) for c in dict_chunk.values()])

        while chunks_remaining > 0:
            printlog(f"manager: chunks remaining: {chunks_remaining}", 0)

            id_analyzer = q_request.get(block=True)

            path_current = dict_analyzer[id_analyzer]

            # if the file is marked not finished, keep on the current path
            if dict_rawstatus[path_current] == "not finished":
                path_used = path_current
                msg = "continuing on raw"
            # if the file is marked finished
            else:
                # find the worker counts on unfinished files
                workercounts = {p: list(dict_analyzer.values()).count(p) for p in dict_rawstatus if dict_rawstatus[p] != "finished"}

                # assign the first path with fewest workers
                path_used = [p for p in workercounts.keys() if workercounts[p] <= min(workercounts.values())][0]

                # and update worker dict
                dict_analyzer[id_analyzer] = path_used

                msg = "assigned to new raw"

            chunk_out = dict_chunk[path_used].pop(0)
            # if you took the last chunk, mark the file finished
            if len(dict_chunk[path_used]) == 0:
                dict_rawstatus[path_used] = "finished"

            shortpath_raw = clip_name(path_used, dir_audio)
            printlog(f"manager: analyzer {id_analyzer} {msg} {shortpath_raw}, chunk {round(chunk_out[0], 1), round(chunk_out[1], 1)}", 2)

            assignment = (path_used, chunk_out)

            q_analyze[id_analyzer].put(assignment)
            chunks_remaining = sum([len(c) for c in dict_chunk.values()])

        printlog(f"manager: all chunks assigned, queuing terminate signal for analyzers", 2)
        for q in q_analyze:
            q.put("terminate")


    def worker_analyzer(id_analyzer):
        printlog(f"analyzer {id_analyzer}: launching", 1)

        # ready model
        #
        embedder, _ = get_embedder('yamnet')  # config already gotten at start
        model, config = loadup(modelname)

        classes = config['classes']

        if classes_out is not None:
            colnames_out = ["score_" + c for c in classes_out]
        else:
            colnames_out = []

        columns_desired = ['start', 'end', 'class_predicted', 'score_predicted'] + colnames_out

        q_request.put(id_analyzer)
        assignment = q_analyze[id_analyzer].get()

        timer_analysis = Timer()
        while assignment != "terminate":
            timer_analysis.restart()
            path_raw = assignment[0]
            shortpath_raw = clip_name(path_raw, dir_audio)

            time_from = assignment[1][0]
            time_to = assignment[1][1]

            path_out = os.path.splitext(path_raw)[0] + '_s' + str(time_from.__floor__()) + '_buzzchunk.csv'
            path_out = re.sub(dir_audio, dir_out, path_out)

            chunk_duration = time_to - time_from

            printlog(f"analyzer {id_analyzer}: analyzing {shortpath_raw} from {round(time_from, 1)}s to {round(time_to, 1)}s", 1)
            audio_data, sr_native = load_audio(path_raw, time_from, time_to)  # leave resampling to extract_input

            frames = frame_audio(audio_data, framelength, sr_native, framehop=0.5)
            inputs = extract_input(frames, sr_native=sr_native, embedder=embedder, config=config)
            results = model(np.array(inputs))
            output = translate_results(np.array(results), classes, framelength)

            output['start'] = output['start'] + time_from
            output['end'] = output['end'] + time_from

            output = output[columns_desired]

            output.to_csv(path_out, index=False)

            q_request.put(id_analyzer)

            timer_analysis.stop()
            analysis_rate = (chunk_duration / timer_analysis.get_total()).__round__(1)
            printlog(
                f"analyzer {id_analyzer}: analyzed {shortpath_raw} from {round(time_from, 1)}s to {round(time_to, 1)}s in {timer_analysis.get_total()}s (rate: {analysis_rate})",
                1)

            assignment = q_analyze[id_analyzer].get()
            
        printlog(f"analyzer {id_analyzer}: terminating")
        q_log.put('terminate')
        sys.exit(0)

    def worker_logger():
        log_item = q_log.get(block=True)

        workers_running = cpus
        while workers_running > 0:
            if log_item == 'terminate':
                workers_running -= 1
                continue

            file_log = open(path_log, "a")
            file_log.write(log_item)
            file_log.close()
            log_item = q_log.get(block=True)

        # on terminate, clean up chunks
        merge_chunks(dir_out)

        timer_total.stop()
        closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s"

        print(closing_message)
        file_log = open(path_log, "a")
        file_log.write(closing_message)
        file_log.close()

    # Go!
    #
    printlog(
        f"begin analysis \n"
        f"start time: {timer_total.time_start} \n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n"
        f"memory allotment {memory_allot}\n",
        0)

    # launch analysis_process; will wait immediately
    proc_analyzers = []
    for a in range(cpus):
        proc_analyzers.append(
            multiprocessing.Process(target=worker_analyzer, name=f"analysis_proc{a}", args=([a])))
        proc_analyzers[-1].start()
        pass

    proc_logger = multiprocessing.Process(target=worker_logger)
    proc_logger.start()

    proc_manager = multiprocessing.Process(target=worker_manager())
    proc_manager.start()

    # wait for analysis to finish
    proc_logger.join()

    # total hack here; worker_logger is supposed to merge chunks when it exits, but it's occasionally leaving
    # some chunks unmerged. All chunks *should* be written by the time worker_logger merges, because
    # the worker_analyzer writes in its main loop before exiting and queuing terminate.
    merge_chunks(dir_out)


if __name__ == "__main__":
    analyze_batch(modelname='agricultural_01', cpus=2, memory_allot=2, verbosity=2)
