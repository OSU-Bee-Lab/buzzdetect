import os.path
import re
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from buzzcode.tools import get_unique_dirs, loadUp, load_wav_16k_mono
from buzzcode.process import make_chunklist, make_chunk_from_control, make_chunk

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


def analyze_wav(model, classes, wav_path, frameLength=960, frameHop=480):
    audio_data = load_wav_16k_mono(wav_path)
    audio_data_split = tf.signal.frame(audio_data, frameLength * 16, frameHop * 16, pad_end=True, pad_value=0)

    results = []

    for i, data in enumerate(audio_data_split):
        scores, embeddings, spectrogram = yamnet_model(data)
        result = model(embeddings).numpy()

        class_means = result.mean(axis=0)
        predicted_class_index = class_means.argmax()
        inferred_class = classes[predicted_class_index]

        confidence_score = class_means[predicted_class_index]

        results.append(
            {
                "start": (i * frameHop) / 1000,
                "end": ((i * frameHop) + frameLength) / 1000,
                "classification": inferred_class,
                "confidence": confidence_score
            }
        )

    output_df = pd.DataFrame(results)

    return output_df


def analyze_mp3_in_place(model, classes, mp3_in, dir_out=None, chunklength=1, frameLength=960, frameHop=480, threads=5):
    if dir_out is None:
        dir_out = os.path.dirname(mp3_in)

    # make chunk list
    chunklist = make_chunklist(mp3_in, chunklength)
    batches = len(chunklist) / threads
    batches = batches.__ceil__()

    for batch in range(0, batches):
        # for this batch, which chunks need to be calculated?
        batchstart = batch * threads
        if batch != (batches - 1):
            chunknums = list(range(batchstart, batchstart + threads))
        else:
            chunknums = list(range(batchstart, len(chunklist)))

        chunklist_sub = [chunklist[i] for i in chunknums]
        paths_in = mp3_in * threads

        for chunk in chunklist_sub:
            chunk_start = chunk[0]
            chunk_path = re.sub("\.mp3$", "_s" + chunk_start.__str__() + ".wav", mp3_in)

        # hmmm this won't work anymore, but I really should find a way to make it work.
        # # if the result already exists, return None early
        # result_name = re.sub(string=chunk_name, pattern="\.wav$", repl="_results.txt")
        # result_path = os.path.join(result_dir, result_name)
        #
        # if os.path.exists(result_path): # if this chunk has already been analyzed, skip!
        #     continue

        # generate chunk
        make_chunk(chunk, mp3_in, chunk_path)

        # analyze chunkfile
        chunk_analysis = analyze_wav(model=model, classes=classes, wav_path=chunk_path, frameLength=frameLength,
                                     frameHop=frameHop)
        chunk_analysis["start"] = chunk_analysis["start"] + chunk_start
        chunk_analysis["end"] = chunk_analysis["end"] + chunk_start
        # delete where frame out-runs file? chunk with one frame overlaps?

        # write results
        chunk_analysis.to_csv(path_or_buf=result_path, sep="\t", index=False)

        # delete chunkfile
        os.remove(chunk_path)


def analyze_mp3_batch(modelname, directory_in="./audio_in", directory_out="./output", chunklength_hr=1, frameLength=960,
                      frameHop=480, threads=6, ):
    model, classes = loadUp(modelname)

    if not os.path.exists(directory_out):
        os.makedirs(directory_out)

    raw_files = []
    for root, dirs, files in os.walk(directory_in):
        for file in files:
            if file.endswith('.mp3'):
                raw_files.append(os.path.join(root, file))

    dirs = []
    for path in raw_files:
        dirs.append(os.path.dirname(path))

    dirs = list(set(dirs))

    for dir in dirs:
        dir_out = re.sub(string=dir, pattern=directory_in, repl=directory_out)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

    for file in raw_files:
        dir_out = os.path.dirname(re.sub(string=file, pattern=directory_in, repl=directory_out))
        analyze_mp3_in_place(model=model, classes=classes, mp3_in=file, dir_out=dir_out, frameLength=frameLength,
                             frameHop=frameHop, chunklength=chunklength_hr)


# change path in to dir in, read files automatically; change chunking dir to processing
def analyze_multithread(modelname, threads, storage_allot, memory_allot=4, dir_in="./audio_in", dir_out="./output",
                        dir_proc="./processing", chunklength=None):
    model, classes = loadUp(modelname)

    if memory_allot > 16:
        memory_allot = 16
        print("Warning: chosen memory allotment causes overflow errors; memory allotment reduced to 16GB")

    kbps = 256  # audio bit rate
    chunk_limit = (
            memory_allot *  # desired memory utilization in gigabytes
            (1 / 8) *  # expansion factor; convert memory gigabytes to wav gigabytes
            8 *  # convert to gigabits
            (10 ** 6) *  # convert to kilobits
            (1 / kbps) *  # convert to seconds
            (1 / 3600)  # convert to hours
    )  # in hours

    # but should chunklength just be raw size/threads?
    if chunklength is None:
        chunklength = chunk_limit
    else:
        if chunklength > chunk_limit:
            chunklength = chunk_limit
            print("Warning: desired chunk length exceeds memory allotment; reducing chunk length to " + str(
                chunk_limit.__round__(1)) + " hours")

    paths_raw = []
    for root, dirs, files in os.walk(dir_in):
        for file in files:
            if file.endswith('.mp3'):
                paths_raw.append(os.path.join(root, file))

    #
    # Build control ----
    #

    dir_chunk = os.path.join(dir_proc, 'chunks')

    chunkdf_list = []
    for path in paths_raw:
        chunkdf = pd.DataFrame(make_chunklist(path, chunklength=chunklength), columns=["start", "end"])
        chunkdf.insert(0, "path_in", path)
        chunkdf_list.append(chunkdf)

    control = pd.concat(chunkdf_list)
    # PROBLEM: this assumes a 3 char file extension! e.g. .aiff won't work
    control['filetype'] = control['path_in'].apply(lambda x: re.search(".{4}$", x).group(0))

    paths_chunk = []
    paths_out = []
    for filetype, start, path in zip(control['filetype'], control['start'], control['path_in']):
        path_chunk = re.sub(pattern=filetype, repl="_s" + str(start) + ".wav", string=path)
        path_chunk = re.sub(pattern=dir_in, repl=dir_chunk, string=path_chunk)
        paths_chunk.append(path_chunk)

        path_out = re.sub(pattern=".wav$", repl="_buzzdetect.txt", string=path_chunk)
        path_out = re.sub(pattern=dir_chunk, repl=dir_out, string=path_out)
        paths_out.append(path_out)

    control['path_chunk'] = paths_chunk
    control['runtime'] = control['end'] - control['start']
    control['wavsize'] = (control['runtime'] * kbps) * (1 / 8) * (1 / (10 ** 6))
    control['path_out'] = paths_out

    batches = []
    batch_storage = 0
    batch = 0
    thread_current = 1
    for i in list(range(0, len(control))):
        wav_size = control["wavsize"].iloc[i]

        if wav_size > storage_allot:
            # to do: make this error list _all_ files that exceed storage allotment (also...add option to chunk in place?)
            errmess = "Error: predicted size of wav file generated from " + control['path_raw'][
                i] + " exceeds storage allotment"
            sys.exit(errmess)

        if ((batch_storage + wav_size) > storage_allot) or (thread_current > threads):
            batch += 1
            batches.append(batch)
            thread_current = 1
            batch_storage = wav_size
        else:
            batches.append(batch)
            batch_storage += wav_size

        # for testing purposes
        # print(
        #     "thread " + str(thread_current) + " batch size " + str(batch_storage)
        # )

        thread_current += 1

    control['batch'] = batches

    get_unique_dirs(control['path_chunk'])
    get_unique_dirs(control['path_out'])

    for b in list(range(0, max(control['batch']) + 1)):
        control_sub = control[control['batch'] == b]
        make_chunk_from_control(control_sub)

        for r in list(range(0, len(control_sub))):
            row = control_sub.iloc[r]
            analysis = analyze_wav(
                model,
                classes,
                wav_path=row['path_chunk']
            )

            analysis.to_csv(row['path_out'])
            os.remove(row['path_chunk'])

    # delete the processing folder
