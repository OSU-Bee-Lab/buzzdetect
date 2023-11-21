import tensorflow as tf
import pandas as pd
import threading
import queue
import os
import sys
import subprocess
import re
import librosa
import multiprocessing
from datetime import datetime
from buzzcode.tools import loadup, size_to_runtime, clip_name, load_audio, get_yamnet
from buzzcode.conversion import cmd_convert
from buzzcode.chunking import make_chunklist, cmd_chunk


def analyze_wav(model, classes, wav_path, yamnet=None, framelength=960, framehop=480):
    if yamnet is None:
        yamnet = get_yamnet()

    audio_data = load_audio(wav_path)
    audio_data_split = tf.signal.frame(audio_data, framelength * 16, framehop * 16, pad_end=True, pad_value=0)

    results = []

    for i, data in enumerate(audio_data_split):
        scores, embeddings, spectrogram = yamnet(data)
        result = model(embeddings).numpy()

        class_means = result.mean(axis=0)
        predicted_class_index = class_means.argmax()
        inferred_class = classes[predicted_class_index]

        confidence_score = class_means[predicted_class_index]
        bee_score = class_means[classes.index("bee")]

        results.append(
            {
                "start": (i * framehop) / 1000,
                "end": ((i * framehop) + framelength) / 1000,
                "classification": inferred_class,
                "confidence": confidence_score,
                "confidence_bee": bee_score
            }
        )

    output_df = pd.DataFrame(results)

    return output_df

# test params:
# modelname="OSBA"; cpus=4; chunklength=1; dir_raw="./audio_in"; dir_proc = None; dir_out=None; verbosity=2; cleanup=False; conflict_proc="overwrite"; conflict_out="overwrite"
def analyze_multithread(modelname, cpus, chunklength,
                        dir_raw="./audio_in", dir_proc=None, dir_out=None, verbosity=1,
                        cleanup=True, conflict_proc="quit", conflict_out="quit"):
    total_t_start = datetime.now()

    # filesystem preparation
    #
    dir_model = os.path.join("models", modelname)

    if dir_proc is None:
        dir_proc = os.path.join(dir_model, "processing")

    if dir_out is None:
        dir_out = os.path.join(dir_model, "output")

    dir_conv = os.path.join(dir_proc, "conv")
    dir_chunk = os.path.join(dir_proc, "chunk")

    if conflict_proc == "quit" and os.path.isdir(dir_proc):
        conflict_proc = input(
            "processing directory already exists; how would you like to proceed? [skip/overwrite/quit]")
        if conflict_proc == "quit":
            quit("user chose to quit; exiting analysis")

    if conflict_out == "quit" and os.path.isdir(dir_out):
        conflict_out = input("output directory already exists; how would you like to proceed? [skip/overwrite/quit]")
        if conflict_out == "quit":
            quit("user chose to quit; exiting analysis")

    paths_raw = []
    for root, dirs, files in os.walk(dir_raw):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))

    log_timestamp = total_t_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")

    # process control
    #
    chunk_limit = size_to_runtime(3.9) / 3600

    if chunklength > chunk_limit:
        print("desired chunk length produce overflow errors, setting to maximum, " + chunk_limit.__round__(
            1).__str__() + " hours")
        chunklength = chunk_limit

    q_raw = multiprocessing.Queue()
    for path in paths_raw:
        q_raw.put(path)

    q_chunk = multiprocessing.Queue()
    q_log = multiprocessing.Queue()

    n_converters = min(cpus, len(paths_raw))
    sema_converter = multiprocessing.BoundedSemaphore(value=n_converters)

    n_analysisproc = min(cpus, len(paths_raw)) # worth 1
    sema_analysisproc = multiprocessing.BoundedSemaphore(value=n_analysisproc) # needed for logger to quit

    threadsperproc = (len(paths_raw)/n_analysisproc).__ceil__()

    n_analyzers = n_analysisproc*threadsperproc # always spin up double threads even if this excees
    sema_analyzer = multiprocessing.BoundedSemaphore(value=n_analyzers)

    for _ in range(n_converters):  # start with no analyzer semaphores; they will be given by converters
        for _ in range(threadsperproc):
            sema_analyzer.acquire()

    event_analysis = multiprocessing.Event()
    event_log = multiprocessing.Event()

    def printlog(item, item_verb=0):
        time_current = datetime.now()
        q_log.put(f"{time_current} - {item} \n")
        event_log.set()

        if item_verb <= verbosity:
            print(item)

    # worker definition
    #
    def logger(event_log):
        printlog(f"logger initialized", 2)
        event_log.wait()

        while True:
            try:
                log_item = q_log.get(block=True, timeout = 0.5)
            except queue.Empty:
                if q_raw.qsize() == 0 and sema_analysisproc.get_value() == n_analysisproc: # race condition?
                    total_t_delta = datetime.now() - total_t_start

                    closing_message = f"analysis complete; total time: {total_t_delta.total_seconds().__round__(2)}s"
                    print(closing_message)
                    log = open(path_log, "a")
                    log.write(closing_message)
                    log.close()
                    break

                event_log.clear()  # for some reason, you have to clear before waiting
                event_log.wait()  # Wait for the event to be set again
                continue  # re-start the while loop

            if not os.path.exists(path_log):
                os.makedirs(os.path.dirname(path_log), exist_ok=True)
                open(path_log, "x")

            log = open(path_log, "a")
            log.write(log_item)
            log.close()

    def converter(ident):
        sema_converter.acquire()
        printlog(f"converter {ident}: launching", 2)

        while True:
            try:
                path_raw = q_raw.get(block=True, timeout=0.5)
            except queue.Empty:
                printlog(f"converter {ident}: no files in raw queue exiting", 1)
                sema_converter.release()

                printlog(f"converter {ident}: current semaphores {sema_analyzer.get_value()}, releasing {threadsperproc}", 1)
                for _ in range(threadsperproc):
                    sema_analyzer.release() # documentation says I should be able to release 2 at once, but it's throwing exceptions; maybe I can't with bounded?
                event_analysis.set()
                break

            audio_duration = librosa.get_duration(path=path_raw)

            # convert
            #
            path_conv = re.sub(pattern=".mp3$", repl=".wav", string=path_raw)
            path_conv = re.sub(pattern=dir_raw, repl=dir_conv, string=path_conv)
            clipname_conv = clip_name(path_conv, dir_conv)

            if os.path.exists(path_conv) and conflict_proc == "skip":
                printlog(f"converter {ident}: {clipname_conv} already exists; skipping conversion", 1)
            else:
                clipname_raw = clip_name(path_raw, dir_raw)

                os.makedirs(os.path.dirname(path_conv), exist_ok=True)

                conv_cmd = cmd_convert(path_raw, path_conv, verbosity=verbosity)

                printlog(f"converter {ident}: converting {clipname_raw}", 1)
                conv_t_start = datetime.now()
                subprocess.run(conv_cmd)
                conv_t_end = datetime.now()
                conv_t_delta = conv_t_end - conv_t_start
                printlog(
                    f"converter {ident}: converted {audio_duration.__round__(1)}s of audio from {clipname_raw} in {conv_t_delta.total_seconds().__round__(2)}s",
                    2)

            # chunk
            #
            chunk_stub = re.sub(pattern=".wav$", repl="", string=path_conv)
            chunk_stub = re.sub(pattern=dir_conv, repl=dir_chunk, string=chunk_stub)

            chunklist = make_chunklist(filepath=path_conv, chunk_stub=chunk_stub, chunklength=chunklength,
                                       audio_duration=audio_duration)

            for chunk in chunklist:
                path_chunk = chunk[2]
                if os.path.exists(path_chunk) and conflict_proc == "skip":
                    printlog(f"converter {ident}: {clip_name(path_chunk, dir_chunk)} already exists, skipping chunk", 1)
                    q_chunk.put(chunk[2])
                    chunklist.remove(chunk)

            if len(chunklist) > 0:
                os.makedirs(os.path.dirname(chunk_stub), exist_ok=True)
                chunk_cmd = cmd_chunk(path_in=path_conv, chunklist=chunklist, verbosity=verbosity)

                printlog(f"converter {ident}: chunking {clipname_conv}", 1)
                chunk_t_start = datetime.now()
                subprocess.run(chunk_cmd)
                chunk_t_end = datetime.now()
                chunk_t_delta = chunk_t_end - chunk_t_start
                printlog(
                    f"converter {ident}: chunked {len(chunklist)} chunks for {clipname_conv} in {chunk_t_delta.total_seconds().__round__(2)}s",
                    2)

            for chunk in chunklist:
                q_chunk.put(chunk[2])
            event_analysis.set()

            if cleanup:
                printlog(f"converter {ident}: deleting converted audio {path_conv}", 2)
                os.remove(path_conv)

    def analysis_process(event_analysis, ident):
        sema_analysisproc.acquire()

        printlog(f"analysis process {ident}: launching", 1)

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # ready model
        #
        model, classes = loadup(modelname)
        yamnet = get_yamnet()

        def analyzer(event_analysis, tid):
            printlog(f"analysis process {ident}, analyzer {tid}: launching and waiting for semaphores", 1)
            sema_analyzer.acquire()

            printlog(f"analysis process {ident}, analyzer {tid}: waking", 2)

            while True:
                try:
                    path_chunk = q_chunk.get(block=True, timeout=0.25)
                except queue.Empty:
                    if sema_converter.get_value() < n_converters:  # if there are still converters running
                        printlog(f"analysis process {ident}, analyzer {tid}: waiting for more chunks", 2)
                        event_analysis.clear()
                        event_analysis.wait()
                        continue
                    else:
                        printlog(f"analysis process {ident}, analyzer {tid}: chunk queue empty, exiting", 1)
                        sema_analyzer.release()
                        sys.exit(0)

                clipname_chunk = clip_name(path_chunk, dir_chunk)

                printlog(f"analysis process {ident}, analyzer {tid}: launching analysis of {clipname_chunk}", 2)

                path_out = re.sub(pattern=dir_chunk, repl=dir_out, string=path_chunk)
                path_out = re.sub(pattern=".wav$", repl="_buzzdetect.csv", string=path_out)

                if os.path.exists(path_out) and conflict_out == "skip":
                    printlog(f"analysis process {ident}, analyzer {tid}: output file {clip_name(path_out, dir_out)} already exists; skipping analysis",1)
                    continue

                chunk_duration = librosa.get_duration(path=path_chunk)

                # analyze
                #
                printlog(f"analysis process {ident}, analyzer {tid}: analyzing {clipname_chunk}", 1)
                analysis_t_start = datetime.now()
                results = analyze_wav(model=model, classes=classes, wav_path=path_chunk, yamnet=yamnet)
                analysis_t_end = datetime.now()
                analysis_t_delta = analysis_t_end - analysis_t_start
                printlog(
                    f"analysis process {ident}, analyzer {tid}: analyzed {chunk_duration.__round__(1)}s of audio from {clipname_chunk} in {analysis_t_delta.total_seconds().__round__(2)}s",
                    1)

                # write
                #
                os.makedirs(os.path.dirname(path_out), exist_ok=True)

                printlog(f"analysis process {ident}, analyzer {tid}: writing results to {path_out}", 2)
                results.to_csv(path_out)

                # cleanup
                #
                if cleanup:
                    printlog(f"analysis process {ident}, analyzer {tid}: deleting chunk {clipname_chunk}", 2)
                    os.remove(path_chunk)

        threadlist = []
        for t in range(threadsperproc):
            printlog(f"analysis process {ident}: launching analyzer {t}", 1)
            threadlist.append(threading.Thread(target=analyzer, args=[event_analysis, t], name=f"analyzer{ident}_thread{t}"))
            threadlist[-1].start()

        for t in threadlist:
            t.join()

        printlog(f"analysis process {ident}: all threads finished; exiting", 1)
        sema_analysisproc.release()
        sys.exit(0)

    # Go!
    #
    printlog(
        f"begin analysis on {total_t_start} with model {modelname} \n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n"
        f"chunk length in hours: {chunklength}\n"
        f"conflict resolution for process files: {conflict_proc}\n"
        f"conflict resolution for output files: {conflict_out}\n",
        0)

    # launch analysis_process; will wait immediately
    analyzers = []
    for a in range(n_analysisproc):
        analyzers.append(multiprocessing.Process(target=analysis_process, name=f"analysis_proc{a}", args=([event_analysis, a])))
        analyzers[-1].start()
        pass  # Use this loop to keep track of progress

    proc_log = multiprocessing.Process(target=logger, args=(event_log,))
    proc_log.start()

    converters = []
    for c in range(cpus):
        converters.append(multiprocessing.Process(target=converter, name=f"converter{c}", args=([c])))
        converters[-1].start()
        pass  # Use this loop to keep track of progress

    # wait for analysis to finish
    proc_log.join()

if __name__ == "__main__":
    analyze_multithread(modelname="OSBA", cpus=4, chunklength=1, dir_raw="./audio_in", verbosity=2, cleanup=True,
                        conflict_proc="skip", conflict_out="skip")
