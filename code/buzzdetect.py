import argparse
import os
import subprocess

parser = argparse.ArgumentParser(
                    prog='buzzdetect.py',
                    description='A program to detect bee buzzes using ML',
                    epilog='github link here')
                    
parser.add_argument('--action', choices=['train', 'analyze', 'preprocess'], required=True)

#train
parser.add_argument('--saveto', required=False, default="./models/model3")

#anaylze
parser.add_argument('--modelsrc', required=False, default="./models/model1")
parser.add_argument('--type', choices=['single', 'bulk'], required=False)
parser.add_argument('--analyzesrc', required=False)
parser.add_argument('--framelength', required=False, default='1000')
parser.add_argument('--framehop', required=False, default='500')
parser.add_argument('--preprocesspath', required=False)
parser.add_argument('--concat', required=False,default=False)

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
        print(args.analyzesrc)
    except:
        print("Error: Must provide --analyzesrc flag")
        exit()

    from analyzeaudio import *
    
    print(args.modelsrc)
    loadedmodel = loadUp(args.modelsrc)
        
    if (args.type == "bulk"):
    
        for filename in os.listdir(args.analyzesrc):
            f = os.path.join(args.analyzesrc, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)
                #f = open(f"./bulkoutput/{filename}.txt", "x") 
                framez(loadedmodel, f, f"./bulkoutput/{filename}.txt", int(args.framelength), int(args.framehop),bool(args.concat))
    else:
        framez(loadedmodel, args.analyzesrc, f"./singleoutput/{os.path.basename(args.analyzesrc)}.txt", int(args.framelength), int(args.framehop),bool(args.concat))

    
