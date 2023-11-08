import os.path
import re
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from buzzcode.tools import loadUp, load_audio
from buzzcode.chunk import make_chunklist, take_chunks

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


def analyze_wav(model, classes, wav_path, framelength=960, framehop=480):
    audio_data = load_audio(wav_path)
    audio_data_split = tf.signal.frame(audio_data, framelength * 16, framehop * 16, pad_end=True, pad_value=0)

    results = []

    for i, data in enumerate(audio_data_split):
        scores, embeddings, spectrogram = yamnet_model(data)
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
        take_chunks(chunk, mp3_in, chunk_path)

        # analyze chunkfile
        chunk_analysis = analyze_wav(model=model, classes=classes, wav_path=chunk_path, framelength=frameLength,
                                     framehop=frameHop)
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

