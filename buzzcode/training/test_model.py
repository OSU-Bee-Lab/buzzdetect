import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import pandas as pd
import os
import re
import multiprocessing
import numpy as np
from buzzcode.analysis.analyze_directory import loadup, translate_results

# modelname = 'freq_128'; cpus=6; max_per_class=100
def analyze_testfold(modelname, cpus, max_per_class = 100):
    dir_model = os.path.join("models", modelname)
    dir_cache = './training/input_cache_freq'

    dir_out = os.path.join(dir_model, "output_testFold")

    metadata = pd.read_csv(os.path.join(dir_model, "metadata.csv"))

    paths_cache = []
    for c in metadata['classification'].unique():
        sub = metadata[(metadata['classification'] == c) & (metadata['fold'] == 5)]
        if max_per_class is not None:
            n_rows = min(len(sub), max_per_class)
        else:
            n_rows = len(sub)

        paths = sub.sample(n_rows)['path_relative'].to_list()
        paths = [re.sub("^/",'', p) for p in paths]
        paths = [os.path.join(dir_cache, path) for path in paths]
        paths = [os.path.splitext(path)[0] + '.npy' for path in paths]
        paths = [p for p in paths if os.path.exists(p)]
        paths_cache.extend(paths)

    paths_out = [re.sub(dir_cache, dir_out, path_cache) for path_cache in paths_cache]
    paths_out = [re.sub('\\.npy$', '_buzzdetect.csv', p) for p in paths_out]

    dirs_out = set([os.path.dirname(p) for p in paths_out])
    for d in dirs_out:
        os.makedirs(d, exist_ok=True)

    q_assignments = multiprocessing.Queue()
    for a in zip(paths_cache, paths_out):
        q_assignments.put(a)

    for c in range(cpus):
        q_assignments.put(('TERMINATE', 'TERMINATE'))

    def analyzer_cache(worker_id):
        model, config = loadup(modelname)
        classes = config['classes']
        framelength = config['framelength']

        path_cache, path_out = q_assignments.get()
        while path_cache != 'TERMINATE':
            print(f"analyzer {worker_id}: analyzing {path_cache}")
            inputs = np.load(path_cache, allow_pickle=True)
            results = model(inputs)
            output = translate_results(np.array(results), classes, framelength)

            output.to_csv(path_out)

            path_cache, path_out = q_assignments.get()

        print(f"analyzer {worker_id}: finished analyzing")

    proc_analyzers = []
    for c in range(cpus):
        proc_analyzers.append(
            multiprocessing.Process(target=analyzer_cache, name=f"analysis_proc{c}", args=([c])))
        proc_analyzers[-1].start()
        pass


    for a in proc_analyzers:
        a.join()

    print('done!')


if __name__ == "__main__":
    cpus = 8

    modelprompt = input("Input model name; 'all' to validate all: ")

    if modelprompt != "all":
        analyze_testfold(modelname=modelprompt, cpus=cpus)
    else:
        modelnames = os.listdir('./models')
        modelnames.remove('archive')
        for modelname in modelnames:
            analyze_testfold(modelname=modelname, cpus=cpus, max_per_class=400)
