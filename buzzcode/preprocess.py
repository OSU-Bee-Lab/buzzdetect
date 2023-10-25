import os
import re
import subprocess
import librosa

def chunk_directory(directory_raw):
    rawFiles = []
    for root, dirs, files in os.walk(directory_raw):
        for file in files:
            if file.endswith('.mp3'):
                rawFiles.append(os.path.join(root, file))

    for raw in rawFiles:
        chunk_out = re.sub(pattern = "raw audio", repl = "chunked audio", string = raw)
        chunk_raw(raw, chunk_out)

def make_chunklist(audio_path, chunklength):
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

def take_chunk(chunktuple, audio_path, path_out, band_low = 200):
    chunk_start = chunktuple[0]
    chunk_end = chunktuple[1]

    subprocess.call([
        "ffmpeg",
        "-i", audio_path,  # Input file
        "-y",  # overwrite any chunks that didn't get deleted (from early interrupts)
        "-ar", "16000",  # Audio sample rate
        "-ac", "1",  # Audio channels
        "-af", "highpass = f = " + str(band_low),
        "-ss", str(chunk_start),  # Start time
        "-to", str(chunk_end),  # End time
        "-c:a", "pcm_s16le",  # Audio codec
        path_out  # Output path
    ])

def chunk_raw(audio_path, path_out_base, chunklength = 1, chunks_to_process ="all"):
    chunklist = make_chunklist(audio_path, chunklength)

    if chunks_to_process == "all":
        chunks_to_process = list(range(0, len(chunklist)))

    for c in chunks_to_process:
        take_chunk(chunktuple = chunklist[c], audio_path = audio_path, dir_out= path_out_base)