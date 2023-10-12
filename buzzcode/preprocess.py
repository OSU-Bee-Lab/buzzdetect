import os
import re
import math
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

# note: 1hr wav in this format ~172 MB; seems like I should increase that size to ~1 GB
# but maybe I need to balance memory usage per-thread? Divide total memory by total threads?
def chunk_raw(raw_path, path_out_base, chunkLength_hr = 1):
    chunk_length = int(60 * 60 * chunkLength_hr) # in seconds
    audio_length = librosa.get_duration(path = raw_path)
    chunks = list(range(1, math.ceil(audio_length/chunk_length)))
    #raw_path = '\'' + raw_path + '\''

    for i in chunks:
        chunk_start = (i-1) * chunk_length
        chunk_end = chunk_start + chunk_length
        path_out = path_out_base + "_" + chunk_start.__str__() + ".wav"
        print("exporting", path_out)
        subprocess.call([
            "ffmpeg",
            "-i", raw_path,  # Input file
            "-n", # don't overwrite, just error out
            "-ar", "16000",  # Audio sample rate
            "-ac", "1",  # Audio channels
            "-ss", str(chunk_start),  # Start time
            "-to", str(chunk_end),  # End time
            "-c:a", "pcm_s16le",  # Audio codec
            path_out  # Output path
        ])
