import argparse

parser = argparse.ArgumentParser(
    prog='buzzdetect.py',
    description='A program to detect bee buzzes using ML',
    epilog='github.com/OSU-Bee-Lab/BuzzDetect'
)

subparsers = parser.add_subparsers(help='sub-command help', dest='action', required = True)

# analyze
parser_analyze = subparsers.add_parser('analyze', help = 'analyze something')
parser_analyze.add_argument('--modelname', required=True, type = str)
parser_analyze.add_argument('--dir_in', required=False, default = "./audio_in", type = str)
parser_analyze.add_argument('--dir_out', required=False, default = "./output", type = str)
parser_analyze.add_argument('--chunkLength', required=False, default = 1, type = float)
parser_analyze.add_argument('--frameLength', required=False, default = 1500, type = int)
parser_analyze.add_argument('--frameHop', required=False, default = 750, type = int)

# train
parser_train = subparsers.add_parser('train', help = 'train a new model')
parser_train.add_argument('--modelname', required=True, type = str)
parser_train.add_argument('--epochs', required=False, default = 30, type = int)
parser_train.add_argument('--trainingset', required=True, type = str)

# preprocess; work in progress
# parser_preprocess = subparsers.add_parser('preprocess', help = 'convert audio files into the proper format')
# parser_preprocess.add_argument('--preprocesspath', required=False, type = str)

args = parser.parse_args()

if (args.action == "train"):
    print("Training new model " + args.modelname + " with set " + args.trainingset)
    from buzzcode.train import *
    generate_model(args.modelname, args.trainingset, args.epochs)

elif (args.action== "preprocess"):
    print("this action is still a work in progress!")
    exit()

elif (args.action == "analyze"):
    from buzzcode.analyze import *

    print("analyzing audio in " + args.dir_in + " with model " + args.modelname)
    analyze_mp3_batch(modelname = args.modelname, directory_in = args.dir_in, directory_out = args.dir_out, frameLength = args.frameLength, frameHop = args.frameHop, chunklength_hr=args.chunkLength)