import argparse
import sys
from buzzcode.utils import str2bool

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
parser_analyze.add_argument('--cpus', required=False, type=int, default=1)
parser_analyze.add_argument('--memory', required=False, type=float, default=0.5)
parser_analyze.add_argument('--gpu', required=False, type=str, default='false')
parser_analyze.add_argument('--vram', required=False, type=float, default=None)
parser_analyze.add_argument('--detail', required=False, type=str, default='rich')
parser_analyze.add_argument('--dir_audio', required=False, default="./audio_in", type=str)
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
    from buzzcode.analysis.analyze_audio import analyze_batch

    analyze_batch(
        modelname=args.modelname,
        cpus=args.cpus,
        RAM=args.memory,
        gpu=str2bool(args.gpu),
        VRAM=args.vram,
        dir_audio=args.dir_audio,
        result_detail=args.detail,
        verbosity=args.verbosity,
    )
