import multiprocessing

import buzzcode.config as cfg
from buzzcode.analysis.analysis import Analyzer
from buzzcode.analysis.workers import Coordinator


def analyze(
        modelname: str,
        classes_out: list = 'all',
        precision: float = None,
        framehop_prop: float = 1,
        chunklength: float = 200,
        analyzers_cpu: int = 1,
        analyzer_gpu: bool = False,
        n_streamers: int = None,
        stream_buffer_depth: int = None,
        dir_audio: str = cfg.DIR_AUDIO,
        dir_out: str = None,
        verbosity_print: str = 'PROGRESS',
        verbosity_log: str = 'DEBUG',
        log_progress: bool = False,
        q_gui: multiprocessing.Queue = None,
        event_stopanalysis: multiprocessing.Event = None,
):
    """Analyze audio files using a buzz detection model.

    Parameters
    ----------
    modelname : str
        Name of the model to use for analysis (corresponding to the directory name in the model directory)
    classes_out : list, optional
        List of strings corresponding to the names of neurons to output, by default None
        If neurons_out is specified, output values are raw neuron activations.
        Either neurons_out or precision must be specified.
    precision : float, optional
        Float of the precision value of the model to use to call buzzes
        If precision is specified, output values are binary for the buzz class.
        Calling of non-buzz events is not currently supported; if you would like
        to work with non-buzz events (e.g., rain), specify neurons_out instead.
        Either precision or neurons_out must be specified.
    framehop_prop : float, optional
        Float specifying the overlap between frames; framehop_prop=1 creates contiguous frames;
        framehop_prop=0.5 creates frames that overlap by half their length, by default 1.
    chunklength : float, optional
        Length of audio chunks in seconds, by default 200. Try different values to tune for your machine.
    analyzers_cpu : int, optional
        Number of parallel CPU workers, by default 1
    analyzer_gpu : bool, optional
        Whether to launch a GPU worker for analysis, by default False
    n_streamers : int, optional
        The number of simultaneous workers to read audio files, by default None
        If None, attempts to calculate a reasonable number of workers. If you're using GPU,
        you may need to significantly increase this number to keep the GPU fed.
    stream_buffer_depth : int, optional
        How many chunks should the streaming queue hold? If 1, only one streamer can enqueue at a time.
        Max RAM utilizaiton will be (concurrent streamers + stream_buffer_depth) * chunklength, since
        each streamer will hold 1 chunk while waiting to enqueue.
    dir_audio : str, optional
        Directory containing audio files to analyze, by default DIR_AUDIO (see config.py)
    dir_out : str, optional
        Output directory for analysis results, by default None
        If None, creates 'output' subdirectory in model directory
    verbosity_print : str, optional
        Level of verbosity for print statements to console (INFO, DEBUG, WARNING, ERROR), by default 'PROGRESS')
    verbosity_log : str, optional
        Level of verbosity for logging to file (INFO, DEBUG, WARNING, ERROR), by default 'DEBUG'
    log_progress : bool, optional
        Whether or not to log progress statements to file, by default False
        For long analyses with small chunks, this can result in log files megabytes in size.
    q_gui : multiprocessing.Queue, optional
        Queue for passing log messages to GUI, by default None
    event_stopanalysis : multiprocessing.Event, optional
        Event for killing analysis, by default None
        If set, allows external stopping of analysis

    Returns
    -------
    None
        Results are written to output directory as files

    Notes
    -----
    This function processes audio files in parallel using multiple CPU cores
    and optionally GPU. It uses a neural network model to classify sounds
    in the audio files. Results are saved as separate files for each
    analyzed audio chunk.
    """

    coordinator = Coordinator(
        analyzers_cpu=analyzers_cpu,
        analyzer_gpu=analyzer_gpu,
        streamers_total=n_streamers,
        depth=stream_buffer_depth,
        q_gui=q_gui,
        event_analysisdone=event_stopanalysis
    )

    analyzer = Analyzer(
        modelname=modelname,
        classes_out=classes_out,
        precision=precision,
        framehop_prop=framehop_prop,
        chunklength=chunklength,
        dir_audio=dir_audio,
        dir_out=dir_out,
        verbosity_print=verbosity_print,
        verbosity_log=verbosity_log,
        log_progress=log_progress,
        coordinator=coordinator
    )

    analyzer.run()


if __name__ == "__main__":
    analyze(modelname=cfg.DEFAULT_MODEL)
