import os.path
import re
import pandas as pd
from buzzcode.tools import unique_dirs, loadUp, load_audio, size_to_runtime
from buzzcode.chunk import make_chunklist, take_chunks, take_chunks, cmd_chunk
from buzzcode.convert import cmd_convert
from buzzcode.analyze import analyze_wav
from subprocess import Popen



# change path in to dir in, read files automatically; change chunking dir to processing
def analyze_multithread(modelname, threads, dir_raw="./audio_in", dir_out=None, dir_proc="./processing", chunklength=None):
    #
    # Setup - building control dataframes
    #   all dataframes will be built before any processing occurs

    paths_raw = []
    for root, dirs, files in os.walk(dir_raw):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))


    # control_raw
    #
    control_raw = pd.DataFrame(paths_raw, columns=["path_raw"])

    # conversion
    dir_conv = os.path.join(dir_proc, "converted")

    def raw_to_conv(path_raw):
        path_conv = re.sub(pattern=".mp3", repl=".wav",string=path_raw)
        path_conv = re.sub(pattern=dir_raw, repl=dir_conv,string=path_conv)
        return path_conv

    control_raw['path_conv'] = control_raw['path_raw'].apply(lambda x: raw_to_conv(x))
    control_raw['cmd_conv'] = list(map(cmd_convert, control_raw['path_raw'], control_raw['path_conv']))

    # chunking
    dir_chunk = os.path.join(dir_proc, 'chunks')
    chunk_limit = size_to_runtime(3.9)/3600  # greater than 4GB, chunks cause overflow errors; we'll err conservatively
    # is it possible there would be no overflow with riff wavs?

    if chunklength is None:
        chunklength = chunk_limit
        print("Automatically setting chunk length to maximum: " + str(chunk_limit.__round__(1)) + " hours")
    elif chunklength > chunk_limit:
        chunklength = chunk_limit
        print("Desired chunk length causes overflow errors, reducing to " + str(chunk_limit.__round__(1)) + " hours")


    chunkdf_list = []
    chunk_stubs = []
    chunk_cmds = []

    for path_raw, path_conv in zip(control_raw['path_raw'], control_raw['path_conv']):
        chunk_stub = re.sub(pattern=".mp3", repl="",string=path_conv)
        chunk_stub = re.sub(pattern=dir_conv, repl=dir_chunk,string=chunk_stub)
        chunk_stubs.append(chunk_stub)

        chunklist = make_chunklist(filepath=path_raw, chunklength=chunklength) # chunklist from raw because conv doesn't exist yet

        chunk_cmd = cmd_chunk(path_in=path_conv, stub_out=chunk_stub, chunklist=chunklist)
        chunk_cmds.append(chunk_cmd)

        chunkdf = pd.DataFrame(chunklist, columns=["start", "end"])
        chunkdf['chunk_stub'] = chunk_stub
        chunkdf_list.append(chunkdf)

    control_raw['chunk_stub'] = chunk_stubs
    control_raw['cmd_chunk'] = chunk_cmds

    # control_chunk
    #
    control_chunk = pd.concat(chunkdf_list)

    def stub_to_chunk(chunk_stub, start):
        path_chunk = chunk_stub + "_s" + str(start) + ".wav"
        return path_chunk

    control_chunk['path_chunk'] = list(map(stub_to_chunk, control_chunk['chunk_stub'], control_chunk['start']))

    # output
    if dir_out is None:
        dir_out = os.path.join("./models", modelname, "output")

    def chunk_to_out(path_chunk):
        path_out = re.sub(pattern=dir_chunk,repl=dir_out,string=path_chunk)
        path_out = re.sub(pattern=".wav$",repl='_buzzdetect.txt',string=path_out)
        return(path_out)

    control_chunk['path_out'] = control_chunk['path_chunk'].apply(lambda x: chunk_to_out(x))

    #
    # directory preparation
    #
    dirs_conv = unique_dirs(control_raw['path_conv'], make=True)
    dirs_chunk = unique_dirs(control_chunk['path_chunk'], make=True)
    dirs_out = unique_dirs(control_chunk['path_out'], make=True)

    control_raw.to_csv(os.path.join(dir_out, "control_raw.csv"))
    control_chunk.to_csv(os.path.join(dir_out, "control_chunk.csv"))


    #
    # defining workers
    #
    # I would like these to be verbose, but that would necessitate moving away from the control dataframes
    # and to...lists? Where the worker handles the processing of filenames, etc.

    def converter(cmd_conv):
        os.system(cmd_conv)

    def chunker(cmd_chunk):
        os.system(cmd_chunk)

    model, classes = loadUp(modelname)
    def analyzer(path_chunk, path_out):
        results = analyze_wav(model=model, classes=classes, wav_path=path_chunk)
        results.to_csv(path_out)















def graveyard():
    batches_conv = list(range(0, (len(paths_raw) / threads).__ceil__()))
    for batch in batches_conv:
        batch_start = batch * threads
        batch_end = (batch_start + threads) # index is non-inclusive, so it's fine that this appears 1 too many
        batch_commands = control_raw['cmd_conv'][batch_start:batch_end].to_list() # will also drop indices greater than length

        processes = [Popen(cmd, shell=True) for cmd in batch_commands]

        for p in processes:
            p.wait()


    # chunking
    #

    # make chunk dirs
    unique_dirs(re.sub(dir_raw, dir_chunk, control_raw['path_raw']))
    unique_dirs(control_chunk['path_out'])

    batches = list(range(0, (len(control_chunk)/threads).__ceil__()))

    for batch in batches:
        batch_start = batch * threads
        batch_end = (batch_start + threads) - 1

        control_sub = control_chunk[batch_start:batch_end]

        print("chunking files: \n" + str(control_sub['path_chunk']))
        take_chunks(control_sub)

        for r in list(range(0, len(control_sub))):
            row = control_sub.iloc[r]

            print("analyzing chunk " + row['path_chunk'])
            analysis = analyze_wav(
                model,
                classes,
                wav_path=row['path_chunk']
            )

            analysis.to_csv(row['path_out'])
            os.remove(row['path_chunk'])

    # delete the processing folder
