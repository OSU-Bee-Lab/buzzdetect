import pandas as pd
import os
import soundfile as sf
from buzzcode.tools import search_dir
from buzzcode.analyze import analyze_batch

# modelname = "revision_5"; cpus = 6; memory_allot = 6; verbosity=2
# when iterating through models, could avoid startup time by passing analyze_wav different models instead of restarting the whole analyze_testFold function
def analyze_testFold(modelname, cpus, memory_allot, verbosity=1):
    dir_model = os.path.join("models", modelname)
    dir_training = "./training"
    dir_out = os.path.join(dir_model, "output_testFold")

    metadata = pd.read_csv(os.path.join(dir_model, "metadata.csv"))
    paths_training = metadata[metadata['fold'] != 5]['path'].to_list()
    paths_all = search_dir(dir_training, list(sf.available_formats().keys()))

    # return all paths that weren't used in training
    paths_test = list(set(paths_all) - set(paths_training))

    analyze_batch(modelname, cpus, memory_allot, paths_raw=paths_test, dir_out=dir_out, classes_out=[]) # only need the primary category back, I think


if __name__ == "__main__":
    cpus = 8
    modelprompt = input("Input model name; 'all' to validate all: ")

    if modelprompt != "all":
       analyze_testFold(modelname=modelprompt, cpus=cpus, verbosity=1, conflict_out="quit")
    else:
        modelnames = list(os.walk("./models"))[0][1]
        for modelname in modelnames:
            analyze_testFold(modelname=modelname, cpus=cpus, verbosity=1, conflict_out="skip")