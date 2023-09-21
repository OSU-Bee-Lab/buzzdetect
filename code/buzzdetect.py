import argparse
import os
import subprocess

parser = argparse.ArgumentParser(
                    prog='buzzdetect.py',
                    description='A program to detect bee buzzes using ML',
                    epilog='github link here')
                    
parser.add_argument('--action', choices=['train', 'analyze', 'preprocess'], required=True)

# ACTIONS: train or analyze
parser.add_argument('--modelname', required=True)

# ACTION: train
parser.add_argument('--trainingset', required=False, default="all")

# ACTION: analyze
parser.add_argument('--dir_in', required=False)
parser.add_argument('--dir_out', required=False)
parser.add_argument('--framelength', required=False)
parser.add_argument('--framehop', required=False)

# ACTION: preprocess
parser.add_argument('--preprocesspath', required=False)

args = parser.parse_args()

print(args.action)


if (args.action == "train"):
    
    print(args.saveto)
    from generatemodel import *
    save(args.saveto)

elif (args.action == "preprocess"):
    try:
        print(args.preprocesspath)
    except:
        print("need --preprocesspath to be provided")
        exit()
    
    #subprocess.run("module load ffmpeg/4.1.3-static",shell=True)

    for filename in os.listdir(args.preprocesspath):
        if filename.endswith(".wav"):
            input_file = os.path.join(args.preprocesspath, filename)
            output_file = os.path.join(args.preprocesspath, f"{os.path.splitext(filename)[0]}_16bit.wav")
            cmd = f'module load ffmpeg/4.1.3-static && ffmpeg -y -i "{input_file}" -c:a pcm_s16le "{output_file}"'
            subprocess.run(cmd, shell=True)

    
elif (args.action == "analyze"):
   
    try:
        print(args.type)
    except:
        print("Error: Must provide --type flag")
        exit()
        
    try:
        print(args.dir_in)
    except:
        print("Error: Must provide --dir_in flag")
        exit()

    from analyzeaudio import *
    
    print(args.modelname)
    loadedmodel = loadUp(args.modelname)
        
    if (args.type == "bulk"):
    
        for filename in os.listdir(args.dir_in):
            f = os.path.join(args.dir_in, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
                #f = open(f"./bulkoutput/{filename}.txt", "x") 
                framez(loadedmodel, f, f"./bulkoutput/{filename}.txt", int(args.framelength), int(args.framehop),bool(args.concat))
    else:
        framez(loadedmodel, args.dir_in, f"./singleoutput/{os.path.basename(args.dir_in)}.txt", int(args.framelength), int(args.framehop),bool(args.concat))

    
