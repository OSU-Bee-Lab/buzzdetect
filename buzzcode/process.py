import os
import re
import subprocess
import librosa
from subprocess import Popen
from subprocess import list2cmdline
import pandas as pd
import numpy as np

def chunk_directory(directory_raw):
    rawFiles = []
    for root, dirs, files in os.walk(directory_raw):
        for file in files:
            if file.endswith('.mp3'):
                rawFiles.append(os.path.join(root, file))

    for raw in rawFiles:
        chunk_out = re.sub(pattern = "raw audio", repl = "chunked audio", string = raw)
        chunk_raw(raw, chunk_out)

def make_chunklist(audio_path, chunklength, audio_length=None):
    if audio_length is None:
        audio_length = librosa.get_duration(path = audio_path)
    chunklength_s = int(60 * 60 * chunklength) # in seconds

    chunklist = []
    if chunklength_s > audio_length:
        return [(0, audio_length)]

    start_time = 0
    end_time = start_time + chunklength_s

    while end_time < audio_length:
        time_remaining = audio_length - end_time
        if time_remaining < 30:  # if the remaining time is <30 seconds, combine it into the current chunk (avoids miniscule chunks)
            end_time = audio_length

        chunklist.append((start_time, end_time))

        start_time = end_time
        end_time = start_time+chunklength_s

    return chunklist

def take_chunk(chunklist, path_in_list, path_out_list, band_low=200):
    commands = []
    for chunk, path_in, path_out in zip(chunklist, path_in_list, path_out_list):
        command = list2cmdline(
            [
                "ffmpeg",
                "-i", path_in,  # Input file
                "-y",  # overwrite any chunks that didn't get deleted (from early interrupts)
                "-ar", "16000",  # Audio sample rate
                "-ac", "1",  # Audio channels
                "-af", "highpass = f = " + str(band_low),
                "-ss", str(chunk[0]),  # Start time
                "-to", str(chunk[1]),  # End time
                "-c:a", "pcm_s16le",  # Audio codec
                path_out  # Output path
            ]
        )

        commands.append(command)

    processes = [Popen(cmd, shell = True) for cmd in commands]

    for p in processes:
        p.wait()

def make_chunk_from_control(control, band_low = 200):
    commands = []
    for r in list(range(0, len(control))):
        row = control.iloc[r]
        commands.append(make_chunk_command(row['path_in'], row['path_chunk'], (row['start'], row['end']), band_low))

    processes = [Popen(cmd, shell=True) for cmd in commands]

    for p in processes:
        p.wait()
