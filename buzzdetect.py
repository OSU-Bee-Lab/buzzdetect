import argparse
from buzzcode.tools import str2bool

parser = argparse.ArgumentParser(
    prog='buzzdetect.py',
    description='A program to detect bee buzzes using ML',
    epilog='github.com/OSU-Bee-Lab/buzzdetect'
)

subparsers = parser.add_subparsers(help='sub-command help', dest='action', required=True)


# analyze
parser_analyze = subparsers.add_parser('analyze', help='analyze something')
parser_analyze.add_argument('--modelname', required=True, type=str)
parser_analyze.add_argument('--threads', required=True, type=int)
parser_analyze.add_argument('--chunklength', required=True, type=float)
parser_analyze.add_argument('--dir_raw', required=False, default="./audio_in", type=str)
parser_analyze.add_argument('--dir_proc', required=False, default=None, type=str)
parser_analyze.add_argument('--dir_out', required=False, default=None, type=str)
parser_analyze.add_argument('--verbosity', required=False, default=1, type=int)
parser_analyze.add_argument('--cleanup', required=False, default=True, type=str2bool)
parser_analyze.add_argument('--conflict_proc', required=False, default="quit", type=str)
parser_analyze.add_argument('--conflict_out', required=False, default="quit", type=str)

# train
parser_train = subparsers.add_parser('train', help='train a new model')
parser_train.add_argument('--modelname', required=True, type=str)
parser_train.add_argument('--epochs', required=False, default=30, type=int)
parser_train.add_argument('--trainingset', required=True, type=str)

args = parser.parse_args()

if (args.action == "train"):
    print(f"training new model {args.modelname} with set {args.trainingset}")
    from buzzcode.train import *

    generate_model(args.modelname, args.trainingset, args.epochs)

elif (args.action == "analyze"):
    print(f"analyzing audio in {args.dir_raw} with model {args.modelname}")
    from buzzcode.analyze import analyze_multithread

    analyze_multithread(
        modelname=args.modelname,
        threads=args.threads,
        dir_raw=args.dir_raw,
        dir_proc=args.dir_proc,
        dir_out=args.dir_out,
        chunklength=args.chunklength,
        verbosity=args.verbosity,
        cleanup=args.cleanup,
        conflict_proc=args.conflict_proc,
        conflict_out=args.conflict_out
    )
