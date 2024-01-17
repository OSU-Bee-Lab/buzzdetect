import tensorflow as tf
import pandas as pd
import os
import re
import sys
import librosa
import multiprocessing
import numpy as np
import soundfile as sf
from datetime import datetime
from buzzcode.tools import search_dir, load_audio
from buzzcode.tools_tf import loadup, get_yamnet

memory_tf = 0.350  # memory (in GB) required for single tensorflow process
seconds_per_g = 3600/2.4
framelength = 960
framehop = 480
out_suffix = "_buzzdetect.csv"

class_bee = 'ins_buzz_bee'
class_high = 'ins_buzz_high'
class_low = 'ins_buzz_low'


def solve_memory(memory_allot, cpus):
    n_analyzers = min(cpus, (memory_allot/memory_tf).__floor__()) # allow as many workers as you have memory for
    memory_remaining = memory_allot - (memory_tf*n_analyzers)
    memory_perchunk = min(memory_remaining/cpus, 4) # hard limiting max memory per chunk to 4G because I don't know if strange things will happen above that

    chunklength = memory_perchunk * seconds_per_g

    if(chunklength<(framelength/1000)):
        raise ValueError("memory_allot and cpu combination results in illegally small frames")  # illegally smol

    return chunklength, n_analyzers

def get_gaps(range_in, coverage_in):
    coverage_in = sorted(coverage_in)
    gaps = []

    # gap between range start and coverage start
    if coverage_in[0][0] > range_in[0]:  # if the first coverage starts after the range starts
        gaps.append((0, coverage_in[0][0]))

    # gaps between coverages
    for i in range(0, len(coverage_in) - 1):
        out_current = coverage_in[i]
        out_next = coverage_in[i + 1]

        if out_next[0] > out_current[1]:  # if the next coverage starts after this one ends,
            gaps.append((out_current[1], out_next[0]))

    # gap after coverage
    if coverage_in[len(coverage_in) - 1][1] < range_in[1]:  # if the last coverage ends before the range ends
        gaps.append((coverage_in[len(coverage_in) - 1][1], range_in[1]))

    return gaps

def gaps_to_chunklist(gaps_in, chunklength):
    if chunklength < 0.960:
        raise ValueError("chunklength is illegally small")

    chunklist_master = []

    for gap in gaps_in:
        gap_length = gap[1] - gap[0]
        n_chunks = (gap_length/chunklength).__ceil__()
        chunkpoints = np.arange(gap[0], gap[1], chunklength).tolist()

        chunklist_gap = []
        # can probably list comprehend this
        for i in range(len(chunkpoints)-1):
            chunklist_gap.append((chunkpoints[i], chunkpoints[i+1]))

        chunklist_gap.append((chunkpoints[len(chunkpoints) - 1], gap[1]))
        chunklist_master.extend(chunklist_gap)

    return chunklist_master


def get_coverage(path_raw, dir_raw, dir_out):
    raw_base = os.path.splitext(path_raw)[0]
    path_out = re.sub(dir_raw, dir_out, raw_base) + out_suffix

    if not os.path.exists(path_out):
        return []

    out_df = pd.read_csv(path_out)
    out_df.sort_values("start", inplace=True)
    out_df["coverageGroup"] = (out_df["start"] > out_df["end"].shift()).cumsum()
    df_coverage = out_df.groupby("coverageGroup").agg({"start": "min", "end": "max"})

    coverage = list(zip(df_coverage['start'], df_coverage['end']))

    return coverage # list of tuples

# model, classes = loadup("revision_5_reweight1"); yamnet = get_yamnet()
def analyze_data(model, classes, audio_data, yamnet):
    if audio_data.dtype != 'tf.float32':
        audio_data = tf.cast(audio_data, tf.float32)

    audio_data_split = tf.signal.frame(audio_data, framelength * 16, framehop * 16, pad_end=True, pad_value=0)

    results = []

    for i, data in enumerate(audio_data_split):
        yam_scores, yam_embeddings, spectrogram = yamnet(data)
        transfer_scores = model(yam_embeddings).numpy()[0]

        results_frame = {
            "start": (i * framehop) / 1000,
            "end": ((i * framehop) + framelength) / 1000,
            "class_predicted": classes[transfer_scores.argmax()],
            "score_predicted": transfer_scores[transfer_scores.argmax()]
        }

        indices_out = [classes.index(c) for c in classes]
        scorenames = ['score_' + c for c in classes]
        results_frame.update({scorenames[i]: transfer_scores[i] for i in indices_out})

        results.append(results_frame)

    output_df = pd.DataFrame(results)

    return output_df

# add caching of embeddings
# add ability to auto-guess memory use?
# modelname = "revision_5_reweight1"; cpus=4; memory_allot = 3; dir_raw="./audio_in"; dir_out=None; verbosity=1; conflict_out="quit"; classes_out = [class_bee, class_high, class_low]
def analyze_multithread(modelname, cpus, memory_allot, classes_out = [class_bee, class_high, class_low], dir_raw="./audio_in", dir_out=None, verbosity=1):
    timer_total_start = datetime.now()

    dir_model = os.path.join("models", modelname)

    if dir_out is None:
        dir_out = os.path.join(dir_model, "output")

    chunklength, n_analyzers = solve_memory(memory_allot, cpus)

    log_timestamp = timer_total_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)
    log = open(path_log, "x")
    log.close()

    paths_raw = search_dir(dir_raw, list(sf.available_formats().keys()))

    # start logger early and make these exit prints printlogs?
    if len(paths_raw) == 0:
        print(
            f"no compatible audio files found in raw directory {dir_raw} \n"
            f"audio format must be compatible with soundfile module version {sf.__version__} \n"
            "exiting analysis"
        )
        sys.exit(0)

    raws_chunklist = []
    raws_unfinished = []

    for path in paths_raw:
        audio_duration = librosa.get_duration(path=path)

        coverage = get_coverage(path, dir_raw, dir_out)
        if len(coverage) == 0:
            coverage = [(0, 0)]

        gaps = get_gaps((0,audio_duration), coverage)
        chunklist = gaps_to_chunklist(gaps, chunklength)

        if len(chunklist) > 0:
            raws_unfinished.append(path)
            raws_chunklist.append(chunklist)

    if len(raws_unfinished) == 0:
        print(f"all files in {dir_raw} are fully analyzed; exiting analysis")
        sys.exit(0)

    # process control
    #
    dict_chunk = dict(zip(raws_unfinished, raws_chunklist))

    analyzer_ids = list(range(n_analyzers))
    analyzers_per_raw = (n_analyzers/len(raws_unfinished)).__ceil__()
    # if more analyzers than raws, repeat the list, wrapping the assignment back to the start
    dict_analyzer = {i: (raws_unfinished*analyzers_per_raw)[i] for i in analyzer_ids}

    dict_rawstatus = {p: "finished" for p in paths_raw}
    for p in raws_unfinished:
        dict_rawstatus[p] = "not finished"

    colnames_out = ["score_" + c for c in classes_out]
    columns_desired = ['start', 'end', 'class_predicted', 'score_predicted'] + colnames_out

    q_request = multiprocessing.Queue() # holds...what? raw paths and worker names, so each worker can add its request for work?
    q_analyze = [multiprocessing.Queue() for _ in analyzer_ids]
    q_write = multiprocessing.Queue() # holds output data frames and paths for writers to work on
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

            path_snip = re.sub(dir_raw, "", path_used)
            printlog(f"manager: analyzer {id_analyzer} {msg} {path_snip}, chunk {round(chunk_out[0], 1), round(chunk_out[1], 1)}", 2)

            assignment = (path_used, chunk_out)

            q_analyze[id_analyzer].put(assignment)
            chunks_remaining = sum([len(c) for c in dict_chunk.values()])

        printlog(f"manager: all chunks assigned, queuing terminate signal for analyzers", 2)
        for q in q_analyze:
            q.put("terminate")


    def worker_analyzer(id_analyzer):
        # I could now easily add a progress bar or estimated time to completion
        printlog(f"analyzer {id_analyzer}: launching", 1)

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # ready model
        #
        model, classes = loadup(modelname)
        yamnet = get_yamnet()

        q_request.put(id_analyzer)
        assignment = q_analyze[id_analyzer].get()
        while assignment != "terminate":
            path_raw = assignment[0]
            path_clip = re.sub(dir_raw, '', path_raw)
            time_from = assignment[1][0]
            time_to = assignment[1][1]
            chunk_duration = time_to - time_from

            printlog(f"analyzer {id_analyzer}: analyzing {path_clip} from {round(time_from, 1)}s to {round(time_to, 1)}s", 1)

            timer_analysis_start = datetime.now()
            audio_data = load_audio(path_raw, time_from, time_to)
            results = analyze_data(model=model, classes=classes, audio_data=audio_data, yamnet=yamnet)
            timer_analysis_end = datetime.now()

            timer_analysis = timer_analysis_end - timer_analysis_start
            analysis_rate = (chunk_duration / timer_analysis.total_seconds()).__round__(1)
            printlog(
                f"analyzer {id_analyzer}: analyzed {path_clip} from {round(time_from, 1)}s to {round(time_to, 1)}s in {timer_analysis.total_seconds().__round__(2)}s (rate: {analysis_rate})",
                1)


            results['start'] = results['start'] + time_from
            results['end'] = results['end'] + time_from

            results = results[columns_desired]

            q_write.put((path_raw, results))
            q_request.put(id_analyzer)
            assignment = q_analyze[id_analyzer].get()
            
        printlog(f"analyzer {id_analyzer}: terminating")
        q_write.put(("terminate", id_analyzer))  # not super happy with this; feels a bit hacky
        sys.exit(0)

    def worker_writer():
        printlog(f"writer: initialized", 2)

        dirs_raw = set([os.path.dirname(p) for p in paths_raw])
        dirs_out = [re.sub(dir_raw, dir_out, d) for d in dirs_raw]
        for d in dirs_out:
            os.makedirs(d, exist_ok=True)

        status_analyzers = [True for _ in analyzer_ids]
        while True in status_analyzers:
            path_raw, results = q_write.get()

            if path_raw == "terminate":
                status_analyzers[results] = False
                continue

            path_out = os.path.splitext(path_raw)[0] + out_suffix
            path_out = re.sub(dir_raw, dir_out, path_out)

            if os.path.exists(path_out):
                results_written = pd.read_csv(path_out)
                results_updated = pd.concat([results_written, results],axis=0, ignore_index=True)
                results_updated = results_updated.sort_values(by = "start")

                results_updated.to_csv(path_out, index = False)
            else:
                results.to_csv(path_out)

        printlog(f"writer: terminating")
        q_log.put("terminate")

    def worker_logger():
        log_item = q_log.get(block=True)
        while log_item != "terminate":
            log = open(path_log, "a")
            log.write(log_item)
            log.close()
            log_item = q_log.get(block=True)

        timer_total = datetime.now() - timer_total_start
        closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.total_seconds().__round__(1)}s"

        print(closing_message)
        log = open(path_log, "a")
        log.write(closing_message)
        log.close()

    # Go!
    #
    printlog(
        f"begin analysis \n"
        f"start time: {timer_total_start} \n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n"
        f"memory allotment {memory_allot}\n",
        0)

    # launch analysis_process; will wait immediately
    proc_analyzers = []
    for a in range(n_analyzers):
        proc_analyzers.append(
            multiprocessing.Process(target=worker_analyzer, name=f"analysis_proc{a}", args=([a])))
        proc_analyzers[-1].start()
        pass

    proc_logger = multiprocessing.Process(target=worker_logger)
    proc_logger.start()

    proc_writer = multiprocessing.Process(target=worker_writer)
    proc_writer.start()

    proc_manager = multiprocessing.Process(target=worker_manager())
    proc_manager.start()

    # wait for analysis to finish
    proc_logger.join()


if __name__ == "__main__":
    analyze_multithread(modelname="revision_5_reweight1", cpus=3, memory_allot=4, verbosity=2)
