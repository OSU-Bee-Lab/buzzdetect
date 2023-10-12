import os
import re

from pydub import AudioSegment
from pydub.utils import make_chunks

raw = "./localData/Karlan Forrester - Soybean Attractiveness/raw audio/fake.mp3"
filepath_in = "./localData/.wavtest/o.wav"
directory_raw = "./localData/Karlan Forrester - Soybean Attractiveness/raw audio"


def chunk_directory(directory_raw):
    rawFiles = []

    for root, dirs, files in os.walk(directory_raw):
        for file in files:
            if file.endswith('.mp3'):
                rawFiles.append(os.path.join(root, file))

    for raw in rawFiles:
        chunk_out = re.sub(pattern = "raw audio", repl = "chunked audio", string = raw)
        chunk_raw(raw, chunk_out)

# note: 1hr wav in this format ~1.7 GB; seems like a doable default for memory
# but maybe I need to balance memory usage per-thread? Divide total memory by total threads?
def chunk_raw(raw_path, path_out_base, chunkLength_hr = 1):
    chunk_length = int(1000 * 60 * 60 * chunkLength_hr) # in milliseconds

    f = open(raw_path, 'rb')
    audio_bytes = f.read()
    raw_data = AudioSegment(audio_bytes, sample_width=2, frame_rate=16000, channels=1) # Manually specify file characteristics

    chunks = make_chunks(audio_segment= raw_data, chunk_length = chunk_length)

    for i, chunk in enumerate(chunks):
        chunk_start_s = (i * chunk_length)/1000
        path_out = path_out_base + "_" + chunk_start_s.__str__() + ".wav"
        print("exporting", path_out)
        chunk.export(
            out_f = path_out,
            format="wav",
            codec = "pcm_s16le"
        )