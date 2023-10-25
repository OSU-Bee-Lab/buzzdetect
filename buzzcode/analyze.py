import os.path
import tensorflow_hub as hub
import pandas as pd
from buzzcode.tools import *
from buzzcode.preprocess import *

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

def analyze_wav(model, classes, wav_path, frameLength = 960, frameHop = 480):
    audio_data = load_wav_16k_mono(wav_path)
    audio_data_split = tf.signal.frame(audio_data, frameLength*16, frameHop*16, pad_end=True, pad_value=0)

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
                "start" : (i*frameHop)/1000,
                "end" : ((i*frameHop)+frameLength)/1000,
                "classification" : inferred_class,
                "confidence" : confidence_score
            }
        )

    output_df = pd.DataFrame(results)

    return(output_df)

def analyze_mp3_in_place(model, classes, mp3_in, result_dir = None, chunklength_hr = 1, frameLength = 1500, frameHop = 750):
    # make chunk list
    audio_length = librosa.get_duration(path=mp3_in)
    chunk_length = int(60 * 60 * chunklength_hr)  # in seconds
    chunklist = make_chunklist(audio_length, chunk_length)


    if result_dir is None:
        result_dir = os.path.dirname(mp3_in)

    for chunk in chunklist:
        chunk_start = chunk[0]
        chunk_path = re.sub("\.mp3$", "_s" + chunk_start.__str__() + ".wav", mp3_in)
        chunk_name = os.path.basename(chunk_path)

        # if the result already exists, return None early
        result_name = re.sub(string=chunk_name, pattern="\.wav$", repl="_results.txt")
        result_path = os.path.join(result_dir, result_name)

        if os.path.exists(result_path): # if this chunk has already been analyzed, skip!
            continue

        # generate chunk
        take_chunk(chunk, mp3_in, chunk_path)

        # analyze chunkfile
        chunk_analysis = analyze_wav(model = model, classes = classes, wav_path= chunk_path, frameLength = frameLength, frameHop = frameHop)
        chunk_analysis["start"] = chunk_analysis["start"] + chunk_start
        chunk_analysis["end"] = chunk_analysis["end"] + chunk_start
        # delete where frame out-runs file? chunk with one frame overlaps?

        # write results
        chunk_analysis.to_csv(path_or_buf=result_path, sep="\t", index=False)

        # delete chunkfile
        os.remove(chunk_path)

def analyze_mp3_batch(modelname, directory_in ="./audio_in", directory_out ="./output", chunklength_hr = 1, frameLength = 960, frameHop = 480):
    model = loadUp(modelname)

    if not os.path.exists(directory_out):
        os.makedirs(directory_out)

    classes = []

    with open(os.path.join("models/", modelname, "classes.txt"), "r") as file:
        # Use a for loop to read each line from the file and append it to the list
        for line in file:
            # Remove the newline character and append the item to the list
            classes.append(line.strip())

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
        dir_out = os.path.dirname(re.sub(string = file, pattern=directory_in, repl=directory_out))
        analyze_mp3_in_place(model = model, classes=classes, mp3_in = file, result_dir = dir_out, frameLength = frameLength, frameHop = frameHop, chunklength_hr = chunklength_hr)
