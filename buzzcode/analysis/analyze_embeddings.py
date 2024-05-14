from buzzcode.utils import search_dir, Timer, clip_name, setthreads, load_pickle
from buzzcode.analysis.analysis import loadup, translate_results, merge_chunks
from datetime import datetime
import os
import re
import sys
import multiprocessing
import numpy as np
setthreads(1)


#  modelname = "model_general"; cpus=6; dir_embeddings="./localData/embeddings"; dir_out=None; verbosity=1; conflict_out="quit"; paths_embeddings = None
def analyze_batch(modelname, cpus, dir_embeddings="./localData/embeddings", paths_embeddings=None, verbosity=1):
    timer_total = Timer()

    dir_model = os.path.join("./models", modelname)
    dir_out = os.path.join(dir_model, "output")

    log_timestamp = timer_total.time_start.strftime("%Y-%m-%d_%H%M%S")
    path_log = os.path.join(dir_out, f"log {log_timestamp}.txt")
    os.makedirs(os.path.dirname(path_log), exist_ok=True)
    log = open(path_log, "x")
    log.close()

    if paths_embeddings is None:
        paths_embeddings = search_dir(dir_embeddings, ['embeddings'])

    def path_in_to_out(path_in):
        path_out = re.sub('_embeddings', '_buzzdetect.csv', path_in)
        path_out = re.sub(dir_embeddings, dir_out, path_out)

        return path_out

    # start logger early and make these exit prints printlogs?
    if len(paths_embeddings) == 0:
        print(
            f"no embeddings found in embedding directory {dir_embeddings} \n"
            "exiting analysis"
        )
        sys.exit(0)

    paths_embeddings = [p for p in paths_embeddings if not os.path.exists(path_in_to_out(p))]
    if len(paths_embeddings) == 0:
        print(
            f"all embeddings already analyzed; \n"
            "exiting analysis"
        )
        sys.exit(0)

    q_paths = multiprocessing.Queue()
    for path in paths_embeddings:
        q_paths.put(path)
    for c in range(cpus):
        q_paths.put('TERMINATE')

    q_log = multiprocessing.Queue()

    def printlog(item, item_verb=0):
        time_current = datetime.now()
        q_log.put(f"{time_current} - {item} \n")

        if item_verb <= verbosity:
            print(item)

        return item

    def worker_analyzer(id_analyzer):
        printlog(f"analyzer {id_analyzer}: launching", 1)

        # ready model
        #
        model, config_model = loadup(modelname)
        classes = config_model['classes']

        path_embeddings = q_paths.get()
        timer_analysis = Timer()
        while path_embeddings != 'TERMINATE':
            timer_analysis.restart()
            shortpath_embeddings = clip_name(path_embeddings, dir_embeddings)

            path_out = path_in_to_out(path_embeddings)

            printlog(f"analyzer {id_analyzer}: analyzing {shortpath_embeddings}", 1)

            embeddinglist = load_pickle(path_embeddings)
            # issue resulting from /Lily_Johnson - OSPT_Pesticide_Spray_Fastac+Fitness/2023-08-02_ClintonCounty/9/230801_1252_embeddings (I think)
            # hmmm...it's short, but not 0-length. Try seeing if you can figure out why, or try catching ValueError
            embeddings = [d['embeddings'] for d in embeddinglist]
            results = model(np.array(embeddings))
            results = translate_results(np.array(results), classes)

            results.insert(0, 'start', [d['start'] for d in embeddinglist])
            results.insert(1, 'end', results['start'] + config_model['framelength'])

            subdir_out = os.path.dirname(path_out)
            if not os.path.exists(subdir_out):
                os.makedirs(subdir_out, exist_ok=True)  # exist ok True for race condition (could also pre-generate empty dirs)
            results.to_csv(path_out, index=False)

            timer_analysis.stop()
            analysis_rate = ((len(embeddings)*config_model['framelength']) / timer_analysis.get_total()).__round__(1)
            printlog(
                f"analyzer {id_analyzer}: analyzed {len(embeddings)} frames, {len(embeddings)*config_model['framelength']} seconds in {timer_analysis.get_total()}s (rate: {analysis_rate})",
                1)

            path_embeddings = q_paths.get()
            
        printlog(f"analyzer {id_analyzer}: terminating")
        q_log.put('terminate')
        sys.exit(0)

    def worker_logger():
        log_item = q_log.get(block=True)

        workers_running = cpus
        while workers_running > 0:
            if log_item == 'terminate':
                workers_running -= 1
                continue

            file_log = open(path_log, "a")
            file_log.write(log_item)
            file_log.close()
            log_item = q_log.get(block=True)

        # on terminate, clean up chunks
        merge_chunks(dir_out)

        timer_total.stop()
        closing_message = f"{datetime.now()} - analysis complete; total time: {timer_total.get_total()}s"

        print(closing_message)
        file_log = open(path_log, "a")
        file_log.write(closing_message)
        file_log.close()

    # Go!
    #
    printlog(
        f"begin analysis of embeddings \n"
        f"start time: {timer_total.time_start} \n"
        f"model: {modelname}\n"
        f"CPU count: {cpus}\n",
        0)

    # launch analysis_process; will wait immediately
    proc_analyzers = []
    for a in range(cpus):
        proc_analyzers.append(
            multiprocessing.Process(target=worker_analyzer, name=f"analysis_proc{a}", args=([a])))
        proc_analyzers[-1].start()
        pass

    proc_logger = multiprocessing.Process(target=worker_logger)
    proc_logger.start()

    # wait for analysis to finish
    proc_logger.join()


if __name__ == "__main__":
    analyze_batch(modelname='model_general', cpus=4, verbosity=2)
