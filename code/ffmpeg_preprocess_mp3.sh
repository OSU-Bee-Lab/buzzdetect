#!/bin/bash
for file in $1/*.mp3; do
 echo $file 
 echo "${file%.*}.wav"
 ffmpeg -i "$file" -vn -ar 16000 -ac 1 "${file%.*}_16kHz.wav"
done;

