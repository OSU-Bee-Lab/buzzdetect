import argparse
import sys


# custom class credit: Steven Bethard
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = argparse.ArgumentParser(
    prog='buzzdetect.py',
    description='A program to train and apply machine learning models for bioacoustics',
    epilog='github.com/OSU-Bee-Lab/buzzdetect'
)

subparsers = parser.add_subparsers(help='sub-command help', dest='action', required=True)

# analyze
parser_analyze = subparsers.add_parser('analyze', help='analyze all audio files in a directory')
parser_analyze.add_argument('--modelname', help='the name of the directory holding the model data', required=True, type=str)
parser_analyze.add_argument('--cpus', required=True, type=int)
parser_analyze.add_argument('--memory', required=True, type=float)
parser_analyze.add_argument('--classes', required=False, type=str) # give as...comma-separated list?
parser_analyze.add_argument('--dir_audio', required=False, default="./audio_in", type=str)
parser_analyze.add_argument('--dir_out', required=False, default=None, type=str)
parser_analyze.add_argument('--verbosity', required=False, default=1, type=int)

# train
parser_train = subparsers.add_parser('train', help='train a new model')
parser_train.add_argument('--modelname', help='the name for your new model; created as a directory in ./models/', required=True, type=str)
parser_train.add_argument('--epochs', help='the maximum number of training epochs', required=True, type=int)
parser_train.add_argument('--cpus', help='the number of processes to create if analyzing the test fold', required=False, default=None, type=str)
parser_train.add_argument('--memory', required=False, default=None, type=str)
parser_train.add_argument('--drop', required=False, default=0, type=str)
parser_train.add_argument('--weightpath', required=False, default=None, type=str)
parser_train.add_argument('--testmodel', required=False, default="False", type=str)

args = parser.parse_args()

if (args.action == "train"):
    print(f"training new model {args.modelname} with set {args.trainingset}")
    from buzzcode.train import generate_model

    generate_model(args.modelname, args.trainingset, args.epochs)

elif (args.action == "analyze"):
    print(f"analyzing audio in {args.dir_audio} with model {args.modelname}")
    from buzzcode.analyze_audio import analyze_batch

    analyze_batch(
        modelname=args.modelname,
        cpus=args.cpus,
        memory_allot=args.memory,
        dir_audio=args.dir_audio,
        verbosity=args.verbosity,
    )
