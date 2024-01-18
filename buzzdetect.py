import argparse
from buzzcode.tools import str2bool

parser = argparse.ArgumentParser(
    prog='buzzdetect.py',
    description='A program to detect bee buzzes using ML',
    epilog='github.com/OSU-Bee-Lab/buzzdetect'
)

subparsers = parser.add_subparsers(help='sub-command help', dest='action', required=True)

# analyze
parser_analyze = subparsers.add_parser('analyze', help='analyze all audio files in a directory')
parser_analyze.add_argument('--modelname', required=True, type=str)
parser_analyze.add_argument('--cpus', required=True, type=int)
parser_analyze.add_argument('--memory', required=True, type=float)
parser_analyze.add_argument('--classes', required=False, type=str) # give as...comma-separated list?
parser_analyze.add_argument('--dir_raw', required=False, default="./audio_in", type=str)
parser_analyze.add_argument('--dir_out', required=False, default=None, type=str)
parser_analyze.add_argument('--verbosity', required=False, default=1, type=int)

# train
parser_train = subparsers.add_parser('train', help='train a new model')
parser_train.add_argument('--modelname', required=True, type=str)
parser_train.add_argument('--epochs', required=True, type=int)
parser_train.add_argument('--cpus', required=False, default=None, type=str)
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
    print(f"analyzing audio in {args.dir_raw} with model {args.modelname}")
    from buzzcode.analyze import analyze_batch

    analyze_batch(
        modelname=args.modelname,
        cpus=args.cpus,
        memory_allot=args.memory,
        dir_raw=args.dir_raw,
        dir_out=args.dir_out,
        verbosity=args.verbosity,
    )
