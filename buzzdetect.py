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
parser_analyze.add_argument('--cpus', help='the number of parallel processes to create for analysis', required=False, type=int, default=1)
parser_analyze.add_argument('--memory', help='the total memory to allot to analysis, in GB (estimated)', required=False, type=float, default=None)
parser_analyze.add_argument('--dir_raw', help='the directory holding audio files to be analyzed', required=False, default="./audio_in", type=str)
parser_analyze.add_argument('--dir_out', help='the directory to output the analysis results', required=False, default=None, type=str)
parser_analyze.add_argument('--verbosity', help='how detailed the terminal readout should be (integer, 0-2)', required=False, default=1, type=int)

# train
parser_train = subparsers.add_parser('train', help='train a new model')
parser_train.add_argument('--modelname', help='the name for your new model; created as a directory in ./models/', required=True, type=str)
parser_train.add_argument('--metadata', help='the name of the training metadata CSV file in ./training/metadata', required=False, default = 'metadata_raw', type=str)
parser_train.add_argument('--weights', help='the name of the class weights CSV file in ./training/weights', required=False, default=None, type=str)
parser_train.add_argument('--epochs', help='the maximum number of training epochs (early stopping is on)', required=False, default=100, type=int)

args = parser.parse_args()

if args.action == "train":
    print(f"training new model {args.modelname}")
    from buzzcode.train_model import generate_model
    generate_model(modelname=args.modelname, metadata_name=args.metadata, weights_name=args.weights, epochs_in=args.epochs)

elif args.action == "analyze":
    print(f"analyzing audio in {args.dir_raw} with model {args.modelname}")
    from buzzcode.analyze_directory import analyze_batch

    if args.memory is None:
        args.memory = args.cpus

    analyze_batch(
        modelname=args.modelname,
        cpus=args.cpus,
        memory_allot=args.memory,
        dir_raw=args.dir_raw,
        dir_out=args.dir_out,
        verbosity=args.verbosity,
    )
