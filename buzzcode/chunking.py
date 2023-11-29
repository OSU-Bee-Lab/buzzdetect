import os
import librosa
from subprocess import Popen


def make_chunklist(filepath, chunk_stub=None, chunklength=None, audio_duration=None):
    if audio_duration is None:
        audio_duration = librosa.get_duration(path=filepath)

    if chunklength is None:
        chunklength_s = audio_duration
    else:
        chunklength_s = (60 * 60 * chunklength).__ceil__()  # in seconds

    if chunklength_s >= audio_duration:
        if chunk_stub is None:
            return [(0, audio_duration)]
        else:
            return [(0, audio_duration, chunk_stub + "_s0.wav")]

    start_time = 0 - chunklength_s  # a bit odd, but makes the while loop an easier read
    end_time = start_time + chunklength_s

    chunklist = []

    while end_time < audio_duration:
        start_time += chunklength_s
        end_time += chunklength_s

        # avoids miniscule chunks
        if ((audio_duration - end_time) < 30):
            end_time = audio_duration

        chunktuple = (start_time, end_time)
        if chunk_stub is not None:
            chunk_path = chunk_stub + "_s" + start_time.__floor__().__str__() + ".wav"
            chunktuple += (chunk_path,)

        chunklist.append(chunktuple)

    return chunklist


def make_chunk_paths(chunklist, chunk_stub):
    chunk_paths = []

    for chunktuple in chunklist:
        chunk_path = chunk_stub + "_s" + str(chunktuple[0]) + ".wav"
        chunk_paths.append(chunk_path)

    return chunk_paths


def cmd_chunk(path_in, chunklist, convert=False, overwrite=True, verbosity=1, band_low=200):
    extension = os.path.splitext(path_in)[1].lower()
    if extension == ".mp3":
        print("input file is mp3, adding conversion options to ffmpeg call")
        convert = True

        # improvement: automatically detect which conditions are not satisfied
        # also...is there penalty in redundant ffmpeg options? Why not always set them?

    cmdlist = [
        "ffmpeg",
        "-i", path_in
    ]

    if overwrite:
        cmdlist.extend(['-y'])
    else:
        cmdlist.extend(['-n'])

    if verbosity <= 1:
        cmdlist.extend(["-v", "quiet"])

    if verbosity == 1:
        cmdlist.extend(["-stats"])

    for chunktuple in chunklist:
        cmdlet = [
            "-rf64", "never",
            "-ss", str(chunktuple[0]),
            "-to", str(chunktuple[1])
        ]

        if convert is True:
            cmdlet.extend(
                [
                    "-sample_rate", "16000",  # Audio sample rate
                    "-ac", "1",  # Audio channels
                    "-af", "highpass = f = " + str(band_low),
                    "-c:a", "pcm_s16le"  # Audio codec
                ]
            )

        cmdlet.append(chunktuple[2])

        cmdlist.extend(cmdlet)

    return cmdlist


def take_chunks(control, band_low=200):
    commands = []
    for r in list(range(0, len(control))):
        row = control.iloc[r]
        commands.append(cmd_chunk(row['path_in'], row['path_chunk'], (row['start'], row['end']), band_low))

    processes = [Popen(cmd, shell=True) for cmd in commands]

    for p in processes:
        p.wait()
