import argparse

parser = argparse.ArgumentParser(
    prog='buzzdetect.py',
    description='A program to detect bee buzzes using ML',
    epilog='github.com/OSU-Bee-Lab/BuzzDetect'
)

subparsers = parser.add_subparsers(help='sub-command help', dest='action', required=True)

# analyze
parser_analyze = subparsers.add_parser('analyze', help='analyze something')
parser_analyze.add_argument('--modelname', required=True, type=str)
parser_analyze.add_argument('--threads', required=True, type=int)
parser_analyze.add_argument('--dir_raw', required=False, default="./audio_in", type=str)
parser_analyze.add_argument('--dir_out', required=False, default=None, type=str)
parser_analyze.add_argument('--chunklength', required=False, default=1, type=float)
parser_analyze.add_argument('--quiet', required=False, default=True, type=bool)
parser_analyze.add_argument('--cleanup', required=False, default=480, type=bool)
parser_analyze.add_argument('--overwrite', required=False, default="n", type=str)

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
        dir_out=args.dir_out,
        chunklength=args.chunklength,
        quiet=args.quiet,
        cleanup=args.cleanup,
        overwrite=args.overwrite
    )
