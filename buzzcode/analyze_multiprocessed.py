import os.path
import re
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from buzzcode.tools import unique_dirs, loadUp, load_audio, size_to_runtime
from buzzcode.process import make_chunklist, take_chunks, take_chunks, make_conversion_command
from subprocess import Popen

# change path in to dir in, read files automatically; change chunking dir to processing
def analyze_multithread(modelname, threads, dir_raw="./audio_in", dir_out="./output", dir_proc="./processing", chunklength=None):
    model, classes = loadUp(modelname)

    paths_raw = []
    for root, dirs, files in os.walk(dir_raw):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))

    control = pd.DataFrame(paths_raw, columns=["path_raw"])

    # conversion; conversion and chunking will have to happen serially; no easy way around that
    #

    dir_converted = os.path.join(dir_proc, "converted")

    def raw_to_conv(path_raw):
        path_conv = re.sub(pattern=".mp3", repl=".wav",string=path_raw)
        path_conv = re.sub(pattern=dir_raw, repl=dir_converted,string=path_conv)
        return path_conv

    control['path_conv'] = control['path_raw'].apply(lambda x: raw_to_conv(x))
    unique_dirs(control['path_conv'])

    control['command_conv'] = list(map(make_conversion_command, control['path_raw'], control['path_conv']))

    batches_conv = list(range(0, (len(paths_raw) / threads).__ceil__()))
    for batch in batches_conv:
        batch_start = batch * threads
        batch_end = (batch_start + threads) # index is non-inclusive, so it's fine that this appears 1 too many
        batch_commands = control['command_conv'][batch_start:batch_end].to_list() # will also drop indices greater than length

        processes = [Popen(cmd, shell=True) for cmd in batch_commands]

        for p in processes:
            p.wait()


    # chunking
    #
    chunk_limit = size_to_runtime(3.9)/3600  # greater than 4GB, chunks cause overflow errors; we'll err conservatively
    # is it possible there would be no overflow with riff wavs?

    if chunklength is None:
        chunklength = chunk_limit
        print("Automatically setting chunk length to maximum: " + str(chunk_limit.__round__(1)) + " hours")
    elif chunklength > chunk_limit:
        chunklength = chunk_limit
        print("Desired chunk length causes overflow errors, reducing to " + str(chunk_limit.__round__(1)) + " hours")

    dir_chunk = os.path.join(dir_proc, 'chunks')

    def raw_to_chunkcmd(path_raw):
        chunk_stub = re.sub(pattern=".mp3", repl="",string=path_raw)
        chunk_stub = re.sub(pattern=dir_raw, repl=dir_chunk,string=chunk_stub)

        chunklist = make_chunklist(audio_path=path_raw, chunklength=chunklength)
        cmd = make_chunk_command(path_in=path_raw, stub_out=chunk_stub, chunklist=chunklist)

        return cmd


    control['chunk_stub'], control['chunk_cmd'] = control['path_raw'].apply(lambda x: raw_to_chunkcmd(x))

    # make chunk dirs
    unique_dirs(re.sub(dir_raw, dir_chunk, control['path_raw']))
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
