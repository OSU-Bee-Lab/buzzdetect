import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(
                    prog='buzzdetect.py',
                    description='A program to detect bee buzzes using ML',
                    epilog='github.com/OSU-Bee-Lab/BuzzDetect')
                    
parser.add_argument('--action', choices=['train', 'analyze', 'preprocess'], required=True)

# ACTIONS: train or analyze
parser.add_argument('--modelname', required=True)

# ACTION: train
parser.add_argument('--trainingset', required=False, default="all")

# ACTION: analyze
parser.add_argument('--dir_in', required=False, default = "./audio_in")
parser.add_argument('--dir_out', required=False) # this is where I need a subparser; default for analyze should be different than default for train
parser.add_argument('--frameLength', required=False, default = 500)
parser.add_argument('--frameHop', required=False, default = 250)

# ACTION: preprocess
parser.add_argument('--preprocesspath', required=False)

args = parser.parse_args()

sys.path.insert(0, "./code/")
print(args.action)
if (args.action == "train"):
    print("training model " + args.modelname)
    from buzzcode.train import *
    save(args.saveto)

elif (args.action == "preprocess"):
    try:
        print(args.preprocesspath)
    except:
        print("need --preprocesspath to be provided")
        exit()

    for filename in os.listdir(args.preprocesspath):
        if filename.endswith(".wav"):
            input_file = os.path.join(args.preprocesspath, filename)
            output_file = os.path.join(args.preprocesspath, f"{os.path.splitext(filename)[0]}_16bit.wav")
            cmd = f'module load ffmpeg/4.1.3-static && ffmpeg -y -i "{input_file}" -c:a pcm_s16le "{output_file}"'
            subprocess.run(cmd, shell=True)

    
elif (args.action == "analyze"):
    try:
        print(args.dir_in)
    except:
        print("Error: Must provide --dir_in flag")
        exit()

    from buzzcode.analyze import *
    
    print("analyzing audio in " + str(args.dir_in) + " with model" + str(args.modelname))
    analyze_batch(model_name = str(args.modelname), directory_in = str(args.dir_in), directory_out = str(args.dir_out), frameLength = int(args.frameLength), frameHop = int(args.frameHop))