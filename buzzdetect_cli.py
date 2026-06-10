import argparse
import json
import multiprocessing
import os
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _resolve_dir_out(dir_out, modelname):
    """Mirror Analyzer's default: model's output subdir when dir_out is unset."""
    from src import config as cfg
    if dir_out is not None:
        return dir_out
    return os.path.join(cfg.DIR_MODELS, modelname, cfg.SUBDIR_OUTPUT)


def _resolve_classes_out(modelname, classes_out):
    """Expand the 'all' sentinel to the model's full class list (for manifest comparison)."""
    if classes_out != 'all':
        return classes_out
    from src import config as cfg
    config_path = os.path.join(cfg.DIR_MODELS, modelname, 'config_model.json')
    with open(config_path) as f:
        return json.load(f)['classes']


def reconcile_with_manifest(modelname, dir_out, classes_out, precision, framehop_prop):
    """If the output folder already holds results from different settings, show the
    conflicts and offer to adopt the existing settings. Returns the (possibly
    overridden) (modelname, classes_out, precision, framehop_prop). Exits on decline."""
    from src.pipeline.manifest import build_manifest, read_manifest, diff_manifests

    existing = read_manifest(_resolve_dir_out(dir_out, modelname))
    if existing is None:
        return modelname, classes_out, precision, framehop_prop

    resolved_classes = _resolve_classes_out(modelname, classes_out) if precision is None else classes_out
    candidate = build_manifest(modelname, framehop_prop, precision, resolved_classes)
    conflicts = diff_manifests(existing, candidate)
    if not conflicts:
        return modelname, classes_out, precision, framehop_prop

    print("The output folder already contains results from different settings:")
    for c in conflicts:
        print(f"  - {c}")

    resp = input("\nAdopt the existing settings and continue? [y/N]: ").strip().lower()
    if resp not in ('y', 'yes'):
        print("Exiting without analyzing.")
        sys.exit(0)

    if existing['output_mode'] == 'detections':
        return existing['modelname'], 'all', existing['precision'], existing['framehop_prop']
    return existing['modelname'], existing['classes_out'], None, existing['framehop_prop']


def main():
    from src.pipeline.loglevels import loglevels
    verbosity_choices = list(loglevels.keys())

    parser = argparse.ArgumentParser(
        prog='buzzdetect.py',
        description='Analyze audio files using buzz detection machine learning models for bioacoustics',
        epilog='github.com/OSU-Bee-Lab/buzzdetect'
    )

    # Core analysis parameters
    parser.add_argument('--modelname',
                        help='Name of the model to use for analysis (corresponding to the directory name in the model directory)',
                        required=True, type=str)

    parser.add_argument('--classes_out',
                        help='List of strings corresponding to the names of neurons to output. If "all", outputs all classes. If specified, output values are raw neuron activations.',
                        required=False, default='all', type=str, nargs='*')

    parser.add_argument('--precision',
                        help='Float of the precision value of the model to use to call buzzes. If specified, output values are binary for the buzz class. Either precision or classes_out must be specified.',
                        required=False, default=None, type=float)

    # Processing parameters
    parser.add_argument('--framehop_prop',
                        help='Float specifying the overlap between frames; framehop_prop=1 creates contiguous frames; framehop_prop=0.5 creates frames that overlap by half their length',
                        required=False, default=1, type=float)

    parser.add_argument('--chunklength',
                        help='Length of audio chunks in seconds. Try different values to tune for your machine.',
                        required=False, default=200, type=float)

    # Worker configuration
    parser.add_argument('--analyzers_cpu',
                        help='Number of parallel CPU workers',
                        required=False, default=1, type=int)

    parser.add_argument('--analyzers_gpu',
                        help='Number of parallel GPU workers. If using GPU, you probably do not want any CPU analyzers.',
                        required=False, default=0, type=int)

    parser.add_argument('--n_streamers',
                        help='The number of simultaneous workers to read audio files. If unspecified, attempts to calculate a reasonable number of workers. If using GPU, you may need to significantly increase this number to keep the GPU fed.',
                        required=False, default=None, type=int)

    parser.add_argument('--stream_buffer_depth',
                        help='How many chunks should the streaming queue hold? If 1, only one streamer can enqueue at a time. Max RAM utilization will be (concurrent streamers + stream_buffer_depth) * chunklength.',
                        required=False, default=None, type=int)

    # Directory parameters
    parser.add_argument('--dir_audio',
                        help='Directory containing audio files to analyze',
                        required=False, default='./audio_in', type=str)

    parser.add_argument('--dir_out',
                        help='Output directory for analysis results. If unspecified, creates "output" subdirectory in model directory',
                        required=False, default=None, type=str)

    # Logging parameters
    parser.add_argument('--verbosity_print',
                        help='Level of verbosity for print statements to console',
                        required=False, default='PROGRESS', type=str,
                        choices=verbosity_choices)

    parser.add_argument('--verbosity_log',
                        help='Level of verbosity for logging to file',
                        required=False, default='DEBUG', type=str,
                        choices=verbosity_choices)

    parser.add_argument('--log_progress',
                        help='Whether or not to log progress statements to file. For long analyses with small chunks, this can result in log files megabytes in size.',
                        required=False, default=False, type=str2bool)

    args = parser.parse_args()

    from src.analyze import analyze

    # Handle classes_out: argparse with nargs='*' yields a list; collapse the "all" sentinel.
    classes_out = args.classes_out
    if isinstance(classes_out, list) and len(classes_out) == 1 and classes_out[0] == 'all':
        classes_out = 'all'
    elif isinstance(classes_out, str) and classes_out == 'all':
        classes_out = 'all'

    modelname, classes_out, precision, framehop_prop = reconcile_with_manifest(
        modelname=args.modelname,
        dir_out=args.dir_out,
        classes_out=classes_out,
        precision=args.precision,
        framehop_prop=args.framehop_prop,
    )

    analyze(
        modelname=modelname,
        classes_out=classes_out,
        precision=precision,
        framehop_prop=framehop_prop,
        chunklength=args.chunklength,
        analyzers_cpu=args.analyzers_cpu,
        analyzers_gpu=args.analyzers_gpu,
        n_streamers=args.n_streamers,
        stream_buffer_depth=args.stream_buffer_depth,
        dir_audio=args.dir_audio,
        dir_out=args.dir_out,
        verbosity_print=args.verbosity_print,
        verbosity_log=args.verbosity_log,
        log_progress=args.log_progress,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    main()
