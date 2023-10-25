import os.path
import shutil
import re
import pandas as pd
from buzzcode.preprocess import make_chunklist
# 1. ~List all mp3s~
# 2. Make chunklists for all mp3s
# 3. Write all chunklists into one text file
# 4. Make a function that does the following:
# 	i. read the master chunklist
# 	ii. read directories in the "processing lock" directory
# 	iii. read analysis files in the "outputs" directory
# 	iv. pick the first chunkfile that isn't locked
# 	v. create a processing lock directory with a name matching the chunkfile
# 	vi. process the chunk
# 	vii. write the results to an outputs directory
# 	iix. delete the lock directory


def analyze_multi(chunklength=1, dir_in="./audio_in", dir_proc="./processing", dir_out="./output", dir_chunk = "./audio_in"):
    if not os.path.exists(dir_proc):
        os.mkdir(dir_proc)

    raw_paths = []
    for root, dirs, files in os.walk("./audio_in"):
        for file in files:
            if file.endswith('.mp3'):
                raw_paths.append(os.path.join(root, file))

    chunkdf_master_list=[]
    for path in raw_paths:
        chunklist = make_chunklist(path, chunklength)

        chunkpaths = []
        for chunk in chunklist:
            chunktag = "_s"+str(chunk[0])+".mp3"
            chunkpath = re.sub(".mp3", chunktag, path)
            chunkpath = re.sub(dir_in, dir_chunk, chunkpath)
            chunkpaths.append(chunkpath)

        chunkdf = pd.DataFrame(chunklist, columns=("start", "end"))
        chunkdf.insert(0, "path_raw", path)
        chunkdf.insert(1, "path_chunk", chunkpaths)

        chunkdf_master_list.append(chunkdf)

    chunkdf_master=pd.concat(chunkdf_master_list)
    chunkdf_master.to_csv("./processing/chunklist_master.csv", index = False)

    # here's where I need to run my individual workers

    shutil.rmtree(dir_proc) # delete the whole shebang when processing is finished!


def analyze_multi_worker(dir_proc, dir_in, dir_out, dir_chunk):
    chunklist_master = pd.read_csv("./processing/chunklist_master.csv")

    # list all lock dirs
    dir_lock = os.path.join(dir_proc, "locks")
    lock_dirs = []
    for dirpaths, dirnames, filenames in os.walk(dir_lock):
        if not dirnames:
            lock_dirs.append(dirpaths)

    # associate lock dirs with chunks
    lockdf = pd.DataFrame(lock_dirs,columns=["lockpath"])
    lockdf['path_chunk']=lockdf['lockpath'].str.replace(dir_lock, dir_chunk)

    # list all buzzdetect output files
    output_files = []
    for root, dirs, files in os.walk(dir_out):
        for file in files:
            if file.endswith('_results.txt'): # ahh crap I wished I had the outputs named as "_buzzdetect.txt"
                output_files.append(os.path.join(root, file))

    # associate output files with chunks
    outputdf = pd.DataFrame(output_files, columns=["outpath"])
    outputdf['path_chunk'] = outputdf['outpath'].str.replace(dir_out, dir_chunk)
    outputdf['path_chunk'] = outputdf['path_chunk'].str.replace("_results.txt", ".mp3")

    def check_chunk_processable(path_chunk):
        locked = lockdf['path_chunk'].str.contains(path_chunk).sum() > 0
        analyzed = outputdf['path_chunk'].str.contains(path_chunk).sum() > 0

        if (locked == False) & (analyzed == False):
            return True
        else:
            return False

    chunklist_master['processable'] = chunklist_master['path_chunk'].apply(check_chunk_processable)

    chunk_to_process=chunklist_master[chunklist_master.processable == True].iloc[0]

    # then process one chunk (think I need to make this a function in analyze.py