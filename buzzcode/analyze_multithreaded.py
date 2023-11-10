import threading
import time
import queue  # for queue.Empty()
import os
import os.path
import re
import subprocess
import shutil
from buzzcode.analyze import analyze_wav
from buzzcode.tools import loadUp, size_to_runtime
from buzzcode.chunk import make_chunklist, cmd_chunk
from buzzcode.convert import cmd_convert

# multiprocessing.set_start_method('spawn') # needed for tensorflow

# test params:
# modelname="OSBA"; threads=4; dir_raw="./audio_in"; dir_out=None; chunklength=None; quiet=False; cleanup=False; chunklength=0.05; pid=1
# analyze_multithread("OSBA", 4, cleanup=False)
def analyze_multithread(modelname, threads, dir_raw="./audio_in", dir_out=None, chunklength=None, quiet=True, cleanup=True):
    # ready model
    #
    model, classes = loadUp("OSBA")

    # filesystem preparation
    #
    dir_model = os.path.join("models", modelname)
    dir_proc = os.path.join(dir_model, "processing")
    dir_conv = os.path.join(dir_proc, "conv")
    dir_chunk = os.path.join(dir_proc, "chunk")

    if dir_out is None:
        dir_out = os.path.join(dir_model, "output")

    if os.path.isdir(dir_out):
        overwrite = input("Output directory already exists; overwrite results? [y/n]")
        if overwrite.lower() != "y":
            quit("user chose not to overwrite; quitting analysis")

    paths_raw = []
    for root, dirs, files in os.walk(dir_raw):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))

    # process control
    #
    chunk_limit = size_to_runtime(3.9)/3600

    if chunklength is None:
        print("setting chunk length to maximum, " + chunk_limit.__round__(1).__str__() + " hours")
        chunklength = chunk_limit
    elif chunklength > chunk_limit:
        print("desired chunk length produce overflow errors, setting to maximum, " + chunk_limit.__round__(1).__str__() + " hours")
        chunklength = chunk_limit

    tf_threads = 4

    if tf_threads < threads:
        quit("at least 4 available threads are recommended for tensorflow analysis")

    queue_try_tolerance = 3

    q_raw = queue.Queue()
    for path in paths_raw:
        q_raw.put(path)

    q_chunk = queue.Queue()

    process_sema = threading.BoundedSemaphore(value=threads)

    start_analysis = threading.Event()

    # worker definition
    #
    def worker_convert():
        process_sema.acquire()
        pid = os.getpid()
        print(f"converter {pid}: launching; remaining process semaphores: {process_sema._value}")

        queue_tries = 0

        while True:
            try:
                path_raw = q_raw.get_nowait()
            except queue.Empty:
                queue_tries += 1
                if queue_tries <= queue_try_tolerance:
                    print(f"converter {pid}: queue appears empty; retrying...")
                    time.sleep(0.05)
                    continue
                    # ^ explanation for the retry:
                    # I'm getting unexpected exceptions right at the start of the script where the queue appears empty on the first pop,
                    # even though printing queue size in the except block gives a non-zero size;
                    # it must be that my workers are firing up before the queue fills, for some reason?
                else:
                    print(f"converter {pid}: all raws processed; exiting ")
                    process_sema.release()
                    print(f"converter {pid}: exiting; updated process semaphores: {process_sema._value}")

                    if process_sema._value >= tf_threads:
                        print(f"converter {pid}: sufficient threads for analyzer")
                        start_analysis.set()
                    break

            # convert
            #
            path_conv = re.sub(pattern=".mp3$", repl=".wav", string=path_raw)
            path_conv = re.sub(pattern=dir_raw, repl=dir_conv, string=path_conv)

            os.makedirs(os.path.dirname(path_conv), exist_ok=True)

            conv_cmd = cmd_convert(path_raw, path_conv, quiet=quiet)

            print(f"converter {pid}: converting {path_raw}")
            subprocess.run(conv_cmd)
            print(f"converter {pid}: conversion finished at {path_conv}")

            # chunk
            #
            chunk_stub = re.sub(pattern=".wav$", repl="", string=path_conv)
            chunk_stub = re.sub(pattern=dir_conv, repl=dir_chunk, string=chunk_stub)

            os.makedirs(os.path.dirname(chunk_stub), exist_ok=True)

            chunklist = make_chunklist(filepath=path_conv, chunk_stub=chunk_stub, chunklength=chunklength)

            chunk_cmd = cmd_chunk(path_in=path_conv, chunklist=chunklist, quiet=quiet)

            print(f"converter {pid}: chunking {path_conv}")
            subprocess.run(chunk_cmd)
            print(f"converter {pid}: chunking finished for {path_conv}")

            for c in chunklist:
                q_chunk.put(c[2])

            if cleanup is True:
                print(f"converter {pid}: deleting {path_conv}")
                os.remove(path_conv)

            # Signal the analysis worker to start when the last file is processed
            if process_sema._value >= tf_threads:
                print(f"converter {pid}: sufficient threads for analyzer")
                start_analysis.set()

    def worker_analyze(start_analysis):
        pid = os.getpid()
        print(f"analyzer {pid}: waiting for threads")

        start_analysis.wait()
        print(f"analyzer {pid}: threads and chunks available; launching analysis")

        while True:
            try:
                path_chunk = q_chunk.get_nowait()
            except queue.Empty:
                if q_raw.qsize() > 0:
                    print(f"analyzer {pid}: analysis caught up to chunking; waiting for more chunks")
                    start_analysis.wait()  # Wait for the event to be set again
                    continue  # re-start the while loop
                else:
                    print(f"analyzer {pid}: analysis completed")
                    break

            # analyze
            #
            print(f"analyzer {pid}: analyzing {path_chunk}")
            results = analyze_wav(model=model, classes=classes, wav_path=path_chunk)
            print(f"analyzer {pid}: analysis completed")
            # write
            #
            print(f"analyzer {pid}: building path_out")
            path_out = re.sub(pattern=dir_chunk, repl=dir_out,string=path_chunk)
            path_out = re.sub(pattern=".wav$", repl="_buzzdetect.csv", string=path_out)

            print(f"analyzer {pid}: making directory")
            os.makedirs(os.path.dirname(path_out), exist_ok=True)

            print(f"analyzer {pid}: writing results to {path_out}")
            results.to_csv(path_out)

            # cleanup
            #
            if cleanup is True:
                print(f"analyzer {pid}: deleting {path_chunk}")
                os.remove(path_chunk)

    # Launch analysis
    #

    # launch worker_converts
    for i in range(threads):
        threading.Thread(target=worker_convert).start()

    # launch analysis in a waiting process
    analysis_process = threading.Thread(target=worker_analyze, args=(start_analysis,))
    analysis_process.start()

    # wait for analysis to finish
    analysis_process.join()

    # still need to remove processing dir!
    if cleanup:
        shutil.rmtree(dir_proc)

if __name__ == "__main__":
    analyze_multithread("OSBA", 4, cleanup=True, quiet=False, chunklength = 1)