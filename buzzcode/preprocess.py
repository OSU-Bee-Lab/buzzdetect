# Script to make wav file chunks with Audacity label files from audio MP3 files from SONY recorders  
# using ffmpeg, pydub and scikit-maad
# 
# Be sure ffmpeg is installed
# 
# Specify the directory as a command line argument or run this script on all subdirectories
#
# **Not working** find -type d -exec python3 [PATH TO] audio_downsample_split_id.py {} \;  

import os
import re
import sys

from datetime import datetime, timedelta

from pydub import AudioSegment

from maad import sound, rois
import pandas as pd

# Read working directory from command line option
working_directory = sys.argv[1]
identifier = working_directory.replace("/", "_")
print(identifier)

# Set working directory
os.chdir(working_directory)

# Define chunk length
chunk_length = 3600000 # in milliseconds; 1800000 = 30 min, 3600000 = 60 min, 7200000 = 120 min

# Find raw audio files
files_to_process = [] # Initialize array to hold raw audio files from SONY recorder
files = os.listdir() # Get list of files in directory
print(files)
## Get list of SONY formatted mp3 files
#for file in files:
#    if(re.match("\d+_\d+\\.mp3", file)): # only find files formatted in output from SONY recorders e.g. '211216_1654.mp3'
#        files_to_process.append(file)
## Get pre-processed files downsampled to 16kHz
for file in files:
    if(re.match("\d+_\d+\\_16kHz.wav", file)): # only find downsampled files formatted in output from SONY recorders '211216_1654_16kHz.wav'
    # if(file.startswith("\d") & file.endswith(".mp3")): 
        files_to_process.append(file)

for file in files_to_process:
    start_time = datetime( # Set datetime from raw SONY filename
        int(file[0:2]) + 2000, #Year
        int(file[2:4]), #Month
        int(file[4:6]), #Day
        int(file[7:9]), #Hour
        int(file[9:11]) #Minute
    )
    
    #sound_file = AudioSegment.from_mp3(file) # Load raw sound file
    sound_file = AudioSegment.from_wav(file) # Load raw sound file
    sound_file_length = len(sound_file) # Get length in milliseconds
    audio_chunks = sound_file[::chunk_length]
    for i, chunk in enumerate(audio_chunks):
        chunk_time = start_time + timedelta(milliseconds=chunk_length * i)
        out_wav = identifier + "{0}.wav".format(chunk_time.strftime("%Y%m%d_%H%M%S"))
        out_csv = identifier + "{0}.txt".format(chunk_time.strftime("%Y%m%d_%H%M%S"))
        print("exporting", out_wav)
        #chunk = chunk.set_frame_rate(16000) # Downsample to 16kHz
        chunk.export(out_wav, format="wav") 
        s, fs = sound.load(out_wav)
        buzz_list = rois.find_rois_cwt(s, fs, flims=(370,570), tlen=1, th=0.0001, display = False) # Second harmonic of bee wingbeat 370-570
        try:
            audacity_list = buzz_list.drop(columns=['min_f', 'max_f'])
            audacity_list['label'] = "bee"
            audacity_list.to_csv(out_csv, sep="\t", index=False, header=False)
            print("exporting", out_csv)
        except:
            print("No detections, removing ", out_wav)
            os.remove(out_wav)

#filtered_sound_file = sound_file.low_pass_filter(1000)

#play(sound_file)
# sound_file = AudioSegment.from_mp3("audio 1.mp3")
#audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-40 )
