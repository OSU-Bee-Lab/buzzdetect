import os.path
import re
import pandas as pd
from buzzcode.process import make_chunklist, take_chunk_multi_sox
from buzzcode.analyze import analyze_wav
from buzzcode.tools import loadUp
from buzzcode.tools import make_unique_dirs


def analyze_multi(modelname, threads, chunklength=1, dir_in="./audio_in", dir_out="./output", keep_chunks = False, storage_GB = 32):
    dir_proc = "./processing"
    model, classes = loadUp(modelname)
    os.makedirs(dir_proc, exist_ok=True)

    dir_chunk = os.path.join(dir_proc, "chunks")
    os.makedirs(dir_chunk, exist_ok=True)

    raw_paths = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            if file.endswith('.mp3'):
                raw_paths.append(os.path.join(root, file))

    chunkdf_list=[]
    for path in raw_paths:
        chunklist = make_chunklist(path, chunklength)

        paths_chunk = []
        chunk_exists = []

        paths_out = []
        out_exists = []

        for chunktuple in chunklist:
            chunktag = "_s"+str(chunktuple[0])+".wav"

            path_chunk = re.sub(".mp3", repl=chunktag, string=path)
            path_chunk = re.sub(dir_in, dir_chunk, path_chunk)
            path_chunk_exists = os.path.exists(path_chunk)

            paths_chunk.append(path_chunk)
            chunk_exists.append(path_chunk_exists)

            path_out = re.sub(pattern=dir_chunk,repl=dir_out,string=path_chunk)
            path_out=re.sub(pattern=".mp3",repl=".csv",string=path_out)
            path_out_exists = os.path.exists(path_out)

            paths_out.append(path_out)
            out_exists.append(path_out_exists)


        chunkdf = pd.DataFrame(chunklist, columns=("start", "end"))
        chunkdf.insert(0, "path_raw", path)
        chunkdf.insert(1, "path_chunk", paths_chunk)
        chunkdf.insert(2, "chunk_exists", chunk_exists)
        chunkdf.insert(2, "path_out", paths_out)
        chunkdf.insert(1, "out_exists", out_exists)

        chunkdf_list.append(chunkdf)

    control_master=pd.concat(chunkdf_list)

    make_unique_dirs(control_master['path_chunk'])
    make_unique_dirs(control_master['path_out'])

    batches = list(range(0,(len(control_master)/threads).__ceil__()))

    for batch in batches:
        batch_start = batch * threads
        batch_end = batch_start + threads
        if batch == (len(batches) - 1):
            batch_end = len(control_master)
        chunknums = list(range(batch_start, batch_end))

        control_sub = control_master.iloc[chunknums]

        take_chunk_multi_sox(
            chunklist = list(zip(control_sub['start'], control_sub['end'])),
            path_in_list=control_sub['path_raw'],
            path_out_list=control_sub['path_chunk']
        )

    # shutil.rmtree(dir_proc) # delete the whole shebang when processing is finished!

def worker_chunk():
    # as with R, it's usually bad practice to iterate over pandas rows;
    # I'm going to do it, because I have a stop condition;
    # I don't want to use an apply statement because I don't want to process the whole file
    for i in iterations:
        chunk = master_control.iloc[i]
        output_exists = os.path.exists(chunk['path_out'])
        if output_exists:
            continue

        chunk_exists = os.path.exists(chunk['path_chunk'])
        if chunk_exists:
            continue

        chunking_locked = os.path.exists(chunk['lock_chunking'])
        if chunking_locked:
            continue

        # lock that bad boy
        os.makedirs(chunk['lock_chunking'])

        # chunk that bad boy
        take_chunk(
            chunktuple=(chunk['start'], chunk['end']),
            audio_path=chunk['path_raw'],
            path_out=chunk['path_chunk']
        )

        os.rmdir(chunk['lock_chunking'])

def worker_analyze_cpu():
    for i in iterations:
        chunk = master_control.iloc[i]
        output_exists = os.path.exists(chunk['path_out'])
        if output_exists:
            continue

        chunk_exists = os.path.exists(chunk['path_chunk'])
        if not chunk_exists:
            continue

        analyzing_locked = os.path.exists(chunk['lock_analyzing'])
        if analyzing_locked:
            continue

        # lock!
        os.makedirs(chunk['lock_analyzing'])

        analysis = analyze_wav(
            model=model,
            classes=classes,
            wav_path=chunk['path_chunk']
        )

        analysis.

def worker_analyze_gpu():

def analyze_multi_worker(path_control):
    master_control = pd.read_csv(path_control)
    nchunks = len(master_control)

    # as with R, it's usually bad practice to iterate over pandas rows;
    # I'm going to do it, because I have a stop condition;
    # I don't want to use an apply statement because I don't want to process the whole file
    for i in list(range(0, nchunks)):
        chunk = master_control.iloc[i]
        lock_exists = os.path.exists(chunk['path_lock'])
        output_exists = os.path.exists(chunk['path_out'])

        # if the file is locked or the output already exists, skip
        if lock_exists | output_exists:
            continue

        # lock that bad boy
        os.makedirs(chunk['path_lock'])

        # then process one chunk (think I need to make this a function in analyze.py
        chunk_and_analyze(
            chunktuple=(chunk['start'], chunk['end']),
            path_raw=chunk['path_raw'],
            path_chunk=chunk['path_chunk'],
            path_out=chunk['path_out']
        )

    # then delete lockdir
    os.rmdir(chunk['path_lock'])

def chunk_and_analyze(chunktuple, path_raw, path_chunk, path_out):
    chunk_start = chunktuple[0]
    take_chunk(chunktuple, path_raw, path_chunk, band_low=200)
    chunk_analysis = analyze_wav(model=model, classes=classes, wav_path=path_chunk)
    chunk_analysis["start"] = chunk_analysis["start"] + chunk_start
    chunk_analysis["end"] = chunk_analysis["end"] + chunk_start
    # delete where frame out-runs file? chunk with one frame overlaps?

    # write results
    chunk_analysis.to_csv(path_or_buf=result_path, sep="\t", index=False)

    # delete chunkfile
    os.remove(path_chunk)
