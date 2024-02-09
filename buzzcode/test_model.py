import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import pandas as pd
import os
import re
from buzzcode.analyze_directory import analyze_batch

# modelname = "invProp_stric"; cpus = 6; memory_allot = 6; max_per_class=10
# when iterating through models, could avoid startup time by passing analyze_wav different models instead of restarting the whole analyze_testFold function
def analyze_testFold(modelname, cpus, memory_allot, semantic, max_per_class = None):
    dir_model = os.path.join("models", modelname)
    dir_training = "./training/audio"

    if semantic:
        tag = '_semantic'
    else:
        tag = '_literal'

    dir_out = os.path.join(dir_model, "output_testFold" + tag)

    metadata = pd.read_csv(os.path.join(dir_model, "metadata.csv"))

    paths_test = []
    for c in metadata['classification'].unique():
        sub = metadata[(metadata['classification'] == c) & (metadata['fold'] == 5)]
        if max_per_class is not None:
            n_rows = min(len(sub), max_per_class)
        else:
            n_rows = len(sub)

        paths = sub.sample(n_rows)['path_relative'].to_list()
        paths = [re.sub("^/",'', p) for p in paths]
        paths = [os.path.join(dir_training, path) for path in paths]
        paths = [p for p in paths if os.path.exists(p)]
        paths_test.extend(paths)

    analyze_batch(modelname=modelname, cpus=cpus, memory_allot=memory_allot, dir_raw=dir_training, paths_raw=paths_test, dir_out=dir_out, semantic=semantic)


if __name__ == "__main__":
    cpus = 8
    semantic = False

    modelprompt = input("Input model name; 'all' to validate all: ")

    if modelprompt != "all":
        analyze_testFold(modelname=modelprompt, cpus=cpus, memory_allot=6, semantic=semantic)
    else:
        modelnames = os.listdir('./models')
        modelnames.remove('archive')
        for modelname in modelnames:
            analyze_testFold(modelname=modelname, cpus=cpus, memory_allot=6, semantic=semantic, max_per_class=200)
