import tensorflow_hub as hub
import numpy as np
import os


def get_embedder(embeddername):
    if embeddername.lower() == 'yamnet':
        os.environ["TFHUB_CACHE_DIR"]="./embedders/yamnet"
        yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

        def embedder(data):
            embeddings = yamnet(data)[1]  # element 1 is embeddings
            embeddings = np.array(embeddings)[0, ]  # numpy converts to (1024,1) by default
            return embeddings

        config = dict(
            embeddername='yamnet',
            framelength=0.96,
            samplerate=16000,
            n_embeddings=1024
        )

    elif embeddername.lower() == 'birdnet':
        from tensorflow import lite as tflite

        path_model = 'embedders/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'

        birdnet = tflite.Interpreter(model_path=path_model, num_threads=1)
        birdnet.allocate_tensors()

        # Get input and output tensors.
        input_details = birdnet.get_input_details()
        output_details = birdnet.get_output_details()

        # Get input tensor index
        input_index = input_details[0]["index"]

        #  drops output layer, returning embeddings instead of classifications
        output_index = output_details[0]["index"] - 1

        def embedder(framelist):
            # Reshape input tensor
            birdnet.resize_tensor_input(input_index, [len(framelist), *framelist[0].shape])
            birdnet.allocate_tensors()

            # Extract feature embeddings
            birdnet.set_tensor(input_index, np.array(framelist, dtype="float32"))
            birdnet.invoke()
            embeddings = birdnet.get_tensor(output_index)

            return embeddings

        config = dict(
            embeddername='birdnet',
            framelength=3,
            samplerate=48000,
            n_embeddings=1024
        )

    else:
        print('ERROR: invalid embedder name')
        return

    return embedder, config


def extract_embeddings(frame_data, embedder):
    embeddings = embedder(frame_data)
    return embeddings
