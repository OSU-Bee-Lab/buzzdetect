print("hello from analyzeaudio.py")

import tensorflow as tf
import tensorflow_hub as hub
import re

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

def framez(model, path, outputFilePath, frameLength = 500, frameHop = 250, classes = None):
    if classes is None:
        classes = []

        with open(os.path.join("models/", model_name, "classes.txt"), "r") as file:
            # Use a for loop to read each line from the file and append it to the list
            for line in file:
                # Remove the newline character and append the item to the list
                classes.append(line.strip())

    audio_data = load_wav_16k_mono(path)
    splitted_audio_data = tf.signal.frame(audio_data, frameLength*16, frameHop*16, pad_end=True, pad_value=0)


    with open(outputFilePath,'a') as out:
        out.write('start,end,classification,confidence\n')

    for i, data in enumerate(splitted_audio_data):
        scores, embeddings, spectrogram = yamnet_model(data)
        result = model(embeddings).numpy()

        class_means = result.mean(axis=0)
        predicted_class_index = class_means.argmax()
        inferred_class = classes[predicted_class_index]
        confidence_score = class_means[predicted_class_index]

        with open(outputFilePath,'a') as out:
            out.write(f"{(i*frameHop)/1000},{((i*frameHop)+frameLength)/1000},{inferred_class},{confidence_score}\n")

def framez_batch(model_name, data_directory = "./audio_in", output_directory_base = "./output", frameLength = 500, frameHop = 250):
    model = loadUp(os.path.join("./models/", model_name))
    output_directory = os.path.join(output_directory_base, model_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    classes = []

    with open(os.path.join("models/", model_name, "classes.txt"), "r") as file:
        # Use a for loop to read each line from the file and append it to the list
        for line in file:
            # Remove the newline character and append the item to the list
            classes.append(line.strip())


    file_list = os.listdir(data_directory)

    for file_name in file_list:
        file_path = os.path.join(data_directory, file_name)
        file_output = re.sub(string = file_name, pattern =  r".wav$", repl = ".txt")
        file_path_out = os.path.join(output_directory, file_output)
        framez(model = model, path = file_path, outputFilePath=file_path_out, frameLength = frameLength, frameHop = frameHop, classes = classes)
