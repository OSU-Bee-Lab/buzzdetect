# So much to do...
## Retraining cycle
* Figure out retraining cycle!

## Code
### train.py
* Store training history as python object in model directory (see the pickle package for python)
* Allow to read from different directories

### analyze.py
* Throw out last frame that overruns audio file?
     - This seems less important now that we're using full length data and long frames
* Make system for smart-detecting what data have already been analyzed
    - Read in all available buzzdetect outputs, chunk around those times
    - Or, have user specify overwrite or new file
* Output analysis metadata (what model was used? When was the analysis run? Framerate? Framehop? Other things?)
* Set default output dir to a dir within the model
* Figure out if the 8x expansion of wav size → memory utilization is the result of multithreading (I have 8 threads); does supercomputer see larger expansion?

## Machine Learning Design
* Make ambient_night classification?

## Supercomputing
### Figure out supercomputing
* Run on GPU
* Parallelize
* Learn about scheduling

## Documentation
### README.md
* CLI commands and flags will need to change once we update buzzdetect.py

# So much done!
## Preprocessing pipeline
* Do we need to get Reed's snipper code working?
  - New chunking code working that splits audio into 1h segments, but doesn't downsample or snip out buzzes only; I don't think we have a need to snip out buzzes at this point
*  MP3 → WAV; on supercomputer instead of local?
  -  No longer converting to WAV, but chunking on supercomputer probably makes sense; parallelizing would be great!
* Does tfio.audio.AudioIOTensor` from [this guide](https://www.tensorflow.org/io/tutorials/audio) mean we don't have to worry about trimming down files?
  - I don't think so. Giving up and just working with ffmpeg
 
## Documentation
### README.md
* FileZilla information should be updated when we figure out how to integrate SharePoint
    - Crap. Can't use FileZilla without paying (quite reasonable) license fee


## Code
### train.py
* [done!] Store metadata csv within model directory

### analyze.py
* Output confidence score as probability?
    - Apparently, working with raw confidence scores is normal and acceptable; instead of probability, I need to figure out a confidence cutoff that balances false positives/negatives
* Rewrite to input files from any directory structure (e.g., all WAVs within the input path) and output with a cloned directory structure
    - Done diddly. Takes in all mp3s located in input dir, finds the dirpaths of those mp3s, translates them to dirpaths of the output dir.
 
## Machine Learning Design
* Make sure you're doing biasing correctly! Does the sample from the training set apply to the sample from the field audio?
    - I'm going to call this one done just because I feel comfortable with the biasing at the moment; maybe the biases should shift the prior expectation from rates-in-training to rates-in-field, but rates-in-field can't really be known ahead of time.
* Re-create training set as mp3 instead of downsampled wav
    - Well, it's wav, but wav directly from mp3 with minmal processing (only what's needed for yamnet)
 
## Supercomputing
### Figure out Teams → OSC file transfer
* Are we working with terabytes at once?
  - No. Lily's entire experiment is only 67GB (assuming it's all uploaded on Teams)
 
## Documentation
### README.md
* FileZilla information should be updated when we figure out how to integrate SharePoint
    - Bummer...FileZilla doesn't work with SharePoint unless you pay for Pro. Which honestly might be worth it...
