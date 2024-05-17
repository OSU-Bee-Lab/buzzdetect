import multiprocessing
import librosa
from buzzcode.audio import frame_audio
from buzzcode.utils import search_dir, Timer, save_pickle, setthreads, make_chunklist
from datetime import datetime
import soundfile as sf
import json
import numpy as np
import os
import re
setthreads(1)

# TODO: generalize to embeddername, framehop (and dynamically build yamnet accordingly)
def get_embedder(embeddername):
    if bool(re.search(pattern='yamnet', string=embeddername.lower())):
        import tensorflow as tf

        embedder = tf.keras.models.load_model('./embedders/' + embeddername, compile=False)
        model_config = embedder.get_config()
        frame_config = model_config['layers'][20]['inbound_nodes'][0][3]
        # representation of framehop and framelength are in... integer centiseconds? ms*10
        # I'm going to assume this is the true framehop, since it's in the model, and that the input params get rounded

        framehop = frame_config['frame_step']/frame_config['frame_length']

        config = dict(
            embeddername='yamnet',
            framelength=0.96,
            framehop=framehop,
            samplerate=16000,
            n_embeddings=1024
        )

    # TODO: make work!
    elif embeddername.lower() == 'birdnet':
        raise ValueError('embedding with birdnet is currently broke :(')
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

        config = dict(
            embeddername='birdnet',
            framelength=3,
            samplerate=48000,
            n_embeddings=1024
        )

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

    else:
        print('ERROR: invalid embedder name')
        return

    return embedder, config


def embed_file(path_audio, embedder, config, chunklength, framehop):
    """helper function to apply embedder to larger-than-memory audio data"""
    """ returns list of dicts of format {time:time of frame start, embeddings:embeddings for frame}"""
    audio_duration = librosa.get_duration(path=path_audio)
    track = sf.SoundFile(path_audio)

    can_seek = track.seekable()  # True
    if not can_seek:
        raise ValueError("Input file not compatible with seeking")
    sr_native = track.samplerate

    chunklist = make_chunklist(duration=audio_duration, chunklength=chunklength, chunk_overlap=config['framelength'], chunk_min=chunklength/10)

    output = []
    for time_start, time_end in chunklist:
        start_frame = round(sr_native * time_start)
        frames_to_read = round(sr_native * (time_end - time_start))
        track.seek(start_frame)
        audio_data = track.read(frames_to_read)
        audio_data = librosa.resample(y=audio_data, orig_sr=sr_native, target_sr=config['samplerate'])

        frames = frame_audio(audio_data, framelength=config['framelength'], samplerate=config['samplerate'], framehop=framehop)
        embeddings = embedder(frames)

        times = [(f*framehop*config['framelength']) + time_start for f in range(len(frames))]

        output.extend([{'start': s, 'embeddings': e} for s, e in zip(times, embeddings)])

    return output

# embeddername = 'yamnet'; dir_audio = './localData/raw experiment audio'; cpus=5; dir_out=None
# TODO: save config file in dir_out
# TODO: could embeddings be saved in float16 without precision loss?
def embed_directory(dir_audio='./localData/raw experiment audio', embeddername='yamnet', dir_out=None, cpus=8, framehop=1, chunklength=3600):
    # TODO: remove training slash from dir audio if it exists (messes up dirname)
    if dir_out is None:
        dir_out = os.path.dirname(dir_audio) + '/embeddings'

    paths_audio = search_dir(dir_audio, list(sf.available_formats().keys()))
    paths_out = [re.sub(dir_audio, dir_out, path) for path in paths_audio]
    paths_out = [os.path.splitext(p)[0]+'_embeddings' for p in paths_out]
    paths_audio = [p for i, p in enumerate(paths_audio) if not os.path.exists(paths_out[i])]
    paths_audio = sorted(paths_audio)  # might as well tackle recordings sequentially

    print(f'{datetime.now()} {len(paths_audio)} files left to embed')

    q_audio = multiprocessing.Queue()
    for path in paths_audio:
        q_audio.put(path)

    for c in range(cpus):
        q_audio.put('TERMINATE')

    q_config = multiprocessing.Queue()  # TODO: change this trash way of sharing config!

    def worker_embedder(worker_id):
        embedder, config = get_embedder(embeddername)
        config.update({'framehop': framehop})
        q_config.put(config)

        print(f'{datetime.now()} embedder {worker_id}: launching')
        path_audio = q_audio.get()
        timer_embedder = Timer()
        while path_audio != 'TERMINATE':
            print(f'{datetime.now()} embedder {worker_id}: starting file {re.sub(dir_audio, "", path_audio)}')
            print(f'{datetime.now()} {q_audio.qsize()-cpus} files remaining')  # not actually the correct math when workers are shutting down

            timer_embedder.restart()
            try:
                embeddings = embed_file(path_audio, embedder=embedder, config=config, framehop=framehop, chunklength=chunklength)
                path_out = re.sub(dir_audio, dir_out, os.path.splitext(path_audio)[0] + '_embeddings')
                save_pickle(path_out, embeddings)
                timer_embedder.stop()

                print(f'{datetime.now()} embedder {worker_id}: embedded {re.sub(dir_audio, "", path_audio)} in {round(timer_embedder.get_total() / 60)} minutes')

            except ValueError:
                print(f'{datetime.now()} embedder {worker_id}: sub-frame error in file {re.sub(dir_audio, "", path_audio)}; skipping')

            path_audio = q_audio.get()

        print(f'{datetime.now()} embedder {worker_id}: no files left in queue; quitting')

    proc_embedders = []
    for a in range(cpus):
        proc_embedders.append(
            multiprocessing.Process(target=worker_embedder, name=f"embedder_proc{a}", args=([a])))
        proc_embedders[-1].start()

    path_config = os.path.join(dir_out, 'config.txt')
    if not os.path.exists(path_config):  # TODO: add check if configs are the same, quit embedding if not?
        config = q_config.get()
        with open(path_config, 'x') as f:
            f.write(json.dumps(config))

    for proc in proc_embedders:
        proc.join()
    print('done!')

if __name__ == '__main__':
    embed_directory(cpus=4)


