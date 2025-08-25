import argparse
import multiprocessing


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
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
    parser_analyze.add_argument('--verbosity', required=False, default='INFO', type=str)
    parser_analyze.add_argument('--chunklength', required=False, default=960, type=float, help='length of audio chunks for processing')
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
    parser_train.add_argument('--noise', help='the name of the noise augmentation file', required=False, default=None, type=str)
    parser_train.add_argument('--volume', help='the name of the volume augmentation file', required=False, default=None, type=str)

    args = parser.parse_args()

    if args.action == "train":
        print(f"training new model {args.modelname} with set {args.set}")
        from buzzcode.training.train_model import train_model

        train_model(
            modelname=args.modelname,
            setname=args.set,
            name_translation=args.translation,
            name_noise=args.noise,
            name_volume=args.volume,
            epochs_in=args.epochs
        )

    elif args.action == "analyze":
        from buzzcode.analysis.analyze_audio import analyze

        # Parse classes if provided
        classes_out = ['ins_buzz']  # default value
        if args.keep_all:
            classes_out = 'all'

        analyze(
            modelname=args.modelname,
            classes_out=classes_out,
            chunklength=args.chunklength,
            cpus=args.cpus,
            gpu=args.gpu,
            framehop_prop=args.framehop_prop,
            dir_audio=args.dir_audio,
            verbosity=args.verbosity,
        )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()