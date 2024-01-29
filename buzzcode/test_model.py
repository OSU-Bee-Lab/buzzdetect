import pandas as pd
import os
import re
import soundfile as sf
from buzzcode.utils.tools import search_dir
from buzzcode.analyze_directory import analyze_batch

# modelname = "revision_5_reweight1"; cpus = 6; memory_allot = 6; max_per_class=10
# when iterating through models, could avoid startup time by passing analyze_wav different models instead of restarting the whole analyze_testFold function
def analyze_testFold(modelname, cpus, memory_allot, max_per_class = 100, verbosity=1):
    dir_model = os.path.join("models", modelname)
    dir_training = "./training/audio"
    dir_out = os.path.join(dir_model, "output_testFold")

    metadata = pd.read_csv(os.path.join(dir_model, "metadata.csv"))

    paths_test = []
    for c in metadata['classification'].unique():
        sub = metadata[(metadata['classification'] == c) & (metadata['fold'] == 5)]
        n_rows = min(len(sub), max_per_class)
        paths = sub.sample(n_rows)['path_relative'].to_list()
        paths = [re.sub("^/",'', p) for p in paths]
        paths = [os.path.join(dir_training, path) for path in paths]
        paths_test.extend(paths)

    analyze_batch(modelname=modelname, cpus=cpus, memory_allot=memory_allot, dir_raw=dir_training, paths_raw=paths_test, dir_out=dir_out, classes_out=list(metadata['classification'].unique()))


if __name__ == "__main__":
    cpus = 8
    modelprompt = input("Input model name; 'all' to validate all: ")

    if modelprompt != "all":
        analyze_testFold(modelname=modelprompt, cpus=cpus, memory_allot=6, verbosity=1)
    else:
        modelnames = list(os.walk("./models"))[0][1]
        for modelname in modelnames:
            analyze_testFold(modelname=modelname, cpus=cpus, memory_allot=6, verbosity=1)