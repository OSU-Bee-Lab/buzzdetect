import tensorflow_hub as hub
import numpy as np
import os


def get_embedder(embeddername):
    if embeddername.lower() == 'yamnet':
        os.environ["TFHUB_CACHE_DIR"]="./embedders/yamnet"
        yamnet = hub.load(handle='https://tfhub.dev/google/yamnet/1')

        def embedder(frames):
            """
            :param frames: a list where each element is a numpy array of audio samples
            :return: a list of equal length to the input where each element is a numpy array of embedding values
            """

            # annoyingly, YAMNet doesn't seem to like taking arrays as inputs
            embeddings = [yamnet(frame)[1] for frame in frames]  # element 1 of yamnet output is embeddings
            embeddings = [np.array(e).squeeze() for e in embeddings]
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

        def embedder(frames):
            """
            :param frames: a list where each element is a numpy array of audio samples
            :return: a list of equal length to the input where each element is a numpy array of embedding values
            """

            # Reshape input tensor
            birdnet.resize_tensor_input(input_index, [len(frames), *frames[0].shape])
            birdnet.allocate_tensors()

            # Extract feature embeddings
            birdnet.set_tensor(input_index, np.array(frames, dtype="float32"))
            birdnet.invoke()
            embeddings = birdnet.get_tensor(output_index)
            embeddings = list(embeddings)

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
