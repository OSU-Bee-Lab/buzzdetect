import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    prog='buzzdetect.py',
    description='A program to train and apply machine learning models for bioacoustics',
    epilog='github.com/OSU-Bee-Lab/buzzdetect'
)

subparsers = parser.add_subparsers(help='sub-command help', dest='action', required=True)

# analyze
parser_analyze = subparsers.add_parser('analyze', help='analyze all audio files in audio_in')
parser_analyze.add_argument('--modelname', help='the name of the model directory', required=True,
                            type=str)
parser_analyze.add_argument('--cpus', required=False, default=2, type=int)
parser_analyze.add_argument('--keep_all', required=False, default=False, type=str2bool, help='whether to write the neuron values of every output neuron (True) or just the insect buzzing neuron (False, default)')
parser_analyze.add_argument('--dir_audio', required=False, default="./audio_in", type=str)
parser_analyze.add_argument('--verbosity', required=False, default=1, type=int)
parser_analyze.add_argument('--chunklength', required=False, default=960, type=int, help='length of audio chunks for processing')
parser_analyze.add_argument('--gpu', required=False, default=False, type=str2bool, help='whether to use GPU for processing')
parser_analyze.add_argument('--framehop_prop', required=False, default=1, type=float, help='proportional frame hop, where 1 makes contiguous frames, 0.5 makes half-overlapping frames, etc.')

# train
parser_train = subparsers.add_parser('train', help='train a new model')
parser_train.add_argument('--modelname', help='the name for your new model; created as a directory in ./models/',
                          required=True, type=str)
parser_train.add_argument('--set', help='the name of the training set to train with', required=True, type=str)
parser_train.add_argument('--translation', help='the name of the translation file for renaming raw labels',
                          required=True, type=str)
parser_train.add_argument('--epochs', help='the maximum number of training epochs (but training may stop early)',
                          required=False, default=300, type=int)
parser_train.add_argument('--augment', help='whether or not to use augmented data in training', required=False,
                          default="True", type=str)

args = parser.parse_args()

if args.action == "train":
    print(f"training new model {args.modelname} with set {args.set}")
    from buzzcode.training.train_model import train_model

    train_model(
        modelname=args.modelname,
        setname=args.set,
        name_translation=args.translation,
        epochs_in=args.epochs,
        augment=str2bool(args.augment)
    )


elif args.action == "analyze":
    print(f"analyzing audio in {args.dir_audio} with model {args.modelname}")
    from buzzcode.analysis.analyze_audio import analyze_batch

    # Parse classes if provided
    classes_keep = ['ins_buzz']  # default value
    if args.keep_all:
        classes_keep = 'all'

    analyze_batch(
        modelname=args.modelname,
        classes_keep=classes_keep,
        chunklength=args.chunklength,
        cpus=args.cpus,
        gpu=args.gpu,
        framehop_prop=args.framehop_prop,
        dir_audio=args.dir_audio,
        verbosity=args.verbosity,
    )