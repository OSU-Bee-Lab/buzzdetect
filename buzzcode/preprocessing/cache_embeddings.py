import os
from buzzcode.tools import search_dir
from buzzcode.analysis import load_audio
from buzzcode.analysis.embeddings import get_embedder, extract_embeddings
import soundfile as sf
import tensorflow as tf
import numpy as np
import multiprocessing
import re


# cpus=2; dir_in = './training/audio'; dir_out = './training/embeddings'; conflict='overwrite'; worker_id=0
def cache_embeddings(cpus, dir_in, dir_out, conflict='skip'):
    paths_in = search_dir(dir_in, list(sf.available_formats()))
    paths_out = [re.sub(dir_in, dir_out, path_in) for path_in in paths_in]
    paths_out = [os.path.splitext(path)[0] + '.npy' for path in paths_out]  # change
    dirs_out = set([os.path.dirname(p) for p in paths_out])
    for d in dirs_out:
        os.makedirs(d, exist_ok=True)

    assignments = list(zip(paths_in, paths_out))

    if conflict == 'skip':
        assignments = [assignment for assignment in assignments if not os.path.exists(assignment[1])]

    if len(assignments) == 0:
        print('cached embeddings already exist for every file; quitting')
        return

    q_assignments = multiprocessing.Queue()

    for a in assignments:
        q_assignments.put(a)

    for _ in range(cpus):
        q_assignments.put(('terminate', 'terminate'))

    def worker_embedder(worker_id):
        print(f"embedder {worker_id}: launching")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        yamnet = get_embedder('yamnet')

        assignment = q_assignments.get()
        path_in = assignment[0]
        path_out = assignment[1]

        while path_in != "terminate":
            print(f"{worker_id}: getting embedding for {re.sub(dir_in, '', path_in)}")
            audio_data = load_audio(path_in)
            if len(audio_data) < 0.96*16000:
                print(f"{worker_id}: skipping embedding for {re.sub(dir_in, '', path_in)}; sub-frame audio segment")

                assignment = q_assignments.get()
                path_in = assignment[0]
                path_out = assignment[1]
                continue

            embeddings = extract_embeddings(audio_data, yamnet)

            embeddings = np.asarray(embeddings)[:, 0]  # I don't know why numpy wants to add a dimension

            np.save(path_out, embeddings)
            # save_pickle(path_out, embeddings)

            assignment = q_assignments.get()
            path_in = assignment[0]
            path_out = assignment[1]

        print(f"{worker_id}: terminating")

    embedders = []
    for c in range(cpus):
        embedders.append(
            multiprocessing.Process(target=worker_embedder, name=f"embedder_{c}", args=([c])))
        embedders[-1].start()

    for c in range(cpus):
        embedders[c].join()

    print('done embedding!')

if __name__ == '__main__':
    cache_embeddings(8, './training/audio', './training/embeddings', conflict='skip')

