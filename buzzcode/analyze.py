print("hello from analyzeaudio.py")

import tensorflow_hub as hub
import re
import sys
from buzzcode.tools import *

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

def analyze_file(model, path_in, path_out, frameLength = 500, frameHop = 250, classes = None):
    if classes is None:
        classes = []

        with open(os.path.join("models/", model_name, "classes.txt"), "r") as file:
            # Use a for loop to read each line from the file and append it to the list
            for line in file:
                # Remove the newline character and append the item to the list
                classes.append(line.strip())

    audio_data = load_wav_16k_mono(path_in)
    audio_data_split = tf.signal.frame(audio_data, frameLength*16, frameHop*16, pad_end=True, pad_value=0)

    with open(path_out, 'a') as out:
        out.write('start,end,classification,confidence\n')

    for i, data in enumerate(audio_data_split):
        scores, embeddings, spectrogram = yamnet_model(data)
        result = model(embeddings).numpy()

        class_means = result.mean(axis=0)
        predicted_class_index = class_means.argmax()
        inferred_class = classes[predicted_class_index]
        confidence_score = class_means[predicted_class_index]

        with open(path_out, 'a') as out:
            out.write(f"{(i*frameHop)/1000},{((i*frameHop)+frameLength)/1000},{inferred_class},{confidence_score}\n")

def analyze_batch(model_name, directory_in ="./audio_in", directory_out ="./output", frameLength = 500, frameHop = 250):
    model = loadUp(model_name)
    subdirectory_out = os.path.join(directory_out, model_name)
    if not os.path.exists(subdirectory_out):
        os.makedirs(subdirectory_out)

    classes = []

    with open(os.path.join("models/", model_name, "classes.txt"), "r") as file:
        # Use a for loop to read each line from the file and append it to the list
        for line in file:
            # Remove the newline character and append the item to the list
            classes.append(line.strip())

    file_list = os.listdir(directory_in)

    for file_name in file_list:
        file_path = os.path.join(directory_in, file_name)
        file_output = re.sub(string = file_name, pattern =  r".wav$", repl = ".txt")
        file_path_out = os.path.join(subdirectory_out, file_output)
        analyze_file(model = model, path_in= file_path, path_out=file_path_out, frameLength = frameLength, frameHop = frameHop, classes = classes)
