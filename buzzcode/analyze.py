import tensorflow as tf
import pandas as pd
import threading
import time
import queue
import os
import subprocess
import shutil
import re
import librosa
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

def analyze_multithread(modelname, threads, chunklength, dir_raw="./audio_in", dir_proc = None, dir_out=None, verbosity=1,
                        cleanup=True, conflict_proc="quit", conflict_out="quit"):
    # ready model
    #
    model, classes = loadup(modelname)
    yamnet = get_yamnet()

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
        conflict_proc = input("processing directory already exists; how would you like to proceed? [skip/overwrite/quit]")
        if conflict_proc == "quit":
            quit("user chose to quit; exiting analysis")

    if conflict_out == "quit" and os.path.isdir(dir_out):
        resolve_out = input("output directory already exists; how would you like to proceed? [skip/overwrite/quit]")
        if resolve_out == "quit":
            quit("user chose to quit; exiting analysis")

    paths_raw = []
    for root, dirs, files in os.walk(dir_raw):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))

    path_log = os.path.join(dir_out, "log.txt")

    # process control
    #
    chunk_limit = size_to_runtime(3.9) / 3600

    if chunklength > chunk_limit:
        print("desired chunk length produce overflow errors, setting to maximum, " + chunk_limit.__round__(
            1).__str__() + " hours")
        chunklength = chunk_limit

    tf_threads = 4

    if tf_threads > threads:
        quit(f"at least {tf_threads} available threads are recommended for tensorflow analysis")

    queue_try_tolerance = 2

    q_raw = queue.Queue()
    for path in paths_raw:
        q_raw.put(path)

    q_chunk = queue.Queue()
    q_log = queue.Queue()

    process_sema = threading.BoundedSemaphore(value=threads)

    event_analysis = threading.Event()
    event_log = threading.Event()

    def printlog(item, item_verb=0):
        time_current = datetime.now()
        q_log.put(f"{time_current} - {item} \n")
        event_log.set()

        if item_verb <= verbosity:
            print(item)

    # worker definition
    #
    def worker_logger(event_log):
        printlog(f"logger initialized", 2)
        event_log.wait()

        while True:
            try:
                log_item = q_log.get_nowait()
            except queue.Empty:
                if q_raw.qsize() == 0 and not thread_analysis.is_alive():
                    total_t_delta = datetime.now() - total_t_start

                    log = open(path_log, "a")
                    log.write(f"analysis complete; total time: {total_t_delta.total_seconds().__round__(2)}s")
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

    def worker_convert():
        process_sema.acquire()
        tid = threading.get_ident()
        printlog(f"converter {tid}: launching; remaining process semaphores: {process_sema._value}", 2)

        queue_tries = 0

        while True:
            try:
                path_raw = q_raw.get_nowait()
            except queue.Empty:
                queue_tries += 1
                if queue_tries <= queue_try_tolerance:
                    printlog(f"converter {tid}: queue appears empty; retrying {(queue_try_tolerance - queue_tries) + 1} more time", 2)
                    time.sleep(0.1)
                    continue
                    # ^ explanation for the retry:
                    # I'm getting unexpected exceptions right at the start of the script where the queue appears empty on the first pop,
                    # even though printing queue size in the except block gives a non-zero size;
                    # it must be that my workers are firing up before the queue fills, for some reason?
                else:
                    process_sema.release()
                    printlog(f"converter {tid}: no files in raw queue exiting", 1)

                    if process_sema._value >= tf_threads:
                        printlog(f"converter {tid}: sufficient threads for analyzer", 2)
                        event_analysis.set()
                    break

            audio_duration = librosa.get_duration(path=path_raw)

            # convert
            #
            path_conv = re.sub(pattern=".mp3$", repl=".wav", string=path_raw)
            path_conv = re.sub(pattern=dir_raw, repl=dir_conv, string=path_conv)
            clipname_conv = clip_name(path_conv, dir_conv)

            if os.path.exists(path_conv) and conflict_proc == "skip":
                printlog(f"converter {tid}: {clipname_conv} already exists; skipping conversion", 1)
            else:
                clipname_raw = clip_name(path_raw, dir_raw)

                os.makedirs(os.path.dirname(path_conv), exist_ok=True)

                conv_cmd = cmd_convert(path_raw, path_conv, verbosity=verbosity)

                printlog(f"converter {tid}: converting {clipname_raw}", 1)
                conv_t_start = datetime.now()
                subprocess.run(conv_cmd)
                conv_t_end = datetime.now()
                conv_t_delta = conv_t_end - conv_t_start
                printlog(f"converter {tid}: converted {audio_duration.__round__(1)}s of audio from {clipname_raw} in {conv_t_delta.total_seconds().__round__(2)}s", 2)

            # chunk
            #
            chunk_stub = re.sub(pattern=".wav$", repl="", string=path_conv)
            chunk_stub = re.sub(pattern=dir_conv, repl=dir_chunk, string=chunk_stub)

            chunklist = make_chunklist(filepath=path_conv, chunk_stub=chunk_stub, chunklength=chunklength, audio_duration=audio_duration)

            for chunk in chunklist:
                path_chunk = chunk[2]
                if os.path.exists(path_chunk) and conflict_proc == "skip":
                    printlog(f"converter {tid}: {clip_name(path_chunk, dir_chunk)} already exists, skipping chunk", 1)
                    q_chunk.put(chunk[2])
                    chunklist.remove(chunk)

            if len(chunklist) > 0:
                os.makedirs(os.path.dirname(chunk_stub), exist_ok=True)
                chunk_cmd = cmd_chunk(path_in=path_conv, chunklist=chunklist, verbosity=verbosity)

                printlog(f"converter {tid}: chunking {clipname_conv}", 1)
                chunk_t_start = datetime.now()
                subprocess.run(chunk_cmd)
                chunk_t_end = datetime.now()
                chunk_t_delta = chunk_t_end-chunk_t_start
                printlog(f"converter {tid}: chunked {len(chunklist)} chunks for {clipname_conv} in {chunk_t_delta.total_seconds().__round__(2)}s", 2)

            for chunk in chunklist:
                q_chunk.put(chunk[2])

            if cleanup:
                printlog(f"converter {tid}: deleting converted audio {path_conv}", 2)
                os.remove(path_conv)

            # wake worker_analyze when enough threads are available (and, de-facto, there are chunks available)
            if process_sema._value >= tf_threads:
                printlog(f"converter {tid}: sufficient threads for analyzer", 2)
                event_analysis.set()

    def worker_analyze(event_analysis):
        printlog(f"analyzer: waiting for threads", 1)

        event_analysis.wait()
        printlog(f"analyzer: threads and chunks available; launching analysis", 1)

        while True:
            try:
                path_chunk = q_chunk.get_nowait()
            except queue.Empty:
                if q_raw.qsize() > 0:
                    printlog(f"analyzer: analysis caught up to chunking; waiting for more chunks", 2)
                    event_analysis.clear()  # for some reason, you have to clear before waiting
                    event_analysis.wait()  # Wait for the event to be set again
                    continue  # re-start the while loop
                else:
                    printlog(f"analyzer: all analyses completed", 0)
                    break

            path_out = re.sub(pattern=dir_chunk, repl=dir_out, string=path_chunk)
            path_out = re.sub(pattern=".wav$", repl="_buzzdetect.csv", string=path_out)

            if os.path.exists(path_out) and conflict_out == "skip":
                printlog(f"analyzer: output file {clip_name(path_out, dir_out)} already exists; skipping analysis")
            else:
                clipname_chunk = clip_name(path_chunk, dir_chunk)
                chunk_duration = librosa.get_duration(path=path_chunk)

                # analyze
                #
                printlog(f"analyzer: analyzing {clipname_chunk}", 1)
                analysis_t_start = datetime.now()
                results = analyze_wav(model=model, classes=classes, wav_path=path_chunk, yamnet=yamnet)
                analysis_t_end = datetime.now()
                analysis_t_delta = analysis_t_end - analysis_t_start
                printlog(f"analyzer: analyzed {chunk_duration.__round__(1)}s of audio from {clipname_chunk} in {analysis_t_delta.total_seconds().__round__(2)}s",1)

                # write
                #
                os.makedirs(os.path.dirname(path_out), exist_ok=True)

                printlog(f"analyzer: writing results to {path_out}", 2)
                results.to_csv(path_out)

            # cleanup
            #
            if cleanup:
                printlog(f"analyzer: deleting chunk {clipname_chunk}", 2)
                os.remove(path_chunk)

    # Go!
    #
    total_t_start = datetime.now()

    thread_log = threading.Thread(target=worker_logger, args=(event_log,))
    thread_log.start()

    printlog(f"begin analysis on {datetime.now().strftime('%Y-%d-%m %H:%M:%S')} of {len(paths_raw)} files using model {modelname} and chunk length {chunklength.__round__(2)}h on {threads} threads", 1)

    # launch worker_converts
    for i in range(threads):
        threading.Thread(target=worker_convert).start()

    # launch worker_analyze; will wait immediately
    thread_analysis = threading.Thread(target=worker_analyze, args=(event_analysis,))
    thread_analysis.start()

    # wait for analysis to finish
    thread_log.join()
    thread_analysis.join()


    if cleanup:
        print("deleting processing directory")
        shutil.rmtree(dir_proc)
