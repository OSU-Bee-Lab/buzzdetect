# So much to do...
## Retraining cycle
* Figure out retraining cycle!

## Code
### Overall
* Adjust useage of chunklength to always be in seconds once chunklength is automatically set from memory and cores

### train.py
* Store training history as python object in model directory (see the pickle package for python)
* Allow to read from different directories
* Move building of training to python module

### analyze.py
* Update to use new load_audio function directly with no conversion
* Re-write entire script to use classes and methods instead of functions?
* Make system for smart-detecting what data have already been analyzed
    - Read in all available buzzdetect outputs, chunk around those times
    - Or, have user specify overwrite or new file
* Change cleanup to take arg list, e.g. ["conv", "chunk"]

## Machine Learning Design
* Make ambient_night classification
* Listen to all ambient_day and remove buzzes
* Write custom loss function that penalizes within-buzz error less
* Add dense layer for hierarchical categorization (categorize buzz, then within buzz categorize insect)
* Move away from YAMNet embeddings? YAMNet doesn't have a classification for 

## Supercomputing
### Figure out supercomputing
* Run on GPU

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
