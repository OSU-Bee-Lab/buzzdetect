# So much to do...
## Preprocessing pipeline
* Does tfio.audio.AudioIOTensor` from [this guide](https://www.tensorflow.org/io/tutorials/audio) mean we don't have to worry about trimming down files?
  - Maybe? I can't figure out how to pass the raw audio tensor through the model

## Code
### train.py
* Store training history as python object in model directory (see the pickle package for python)
* Allow to read from different directories

### analyze.py
* Rewrite to input files from any directory structure (e.g., all WAVs within the input path) and output with a cloned directory structure
* Throw out last frame that overruns audio file
* Detect whether output file conflicts ahead of time instead of just appending (maybe just if the dir already exists, make a new dir with a timestamp)
* Output analysis metadata (what model was used? When was the analysis run? Framerate? Framehop? Other things?)

### To build
* MAKE script to create directory structure

## Machine Learning Design
* Make "none"/"ambient" audio in training set?
* Re-create training set as mp3 instead of downsampled wav

## Supercomputing
### Figure out supercomputing
* Run on GPU
* Parallelize
* Learn about scheduling

### Figure out Teams → OSC file transfer
* Should be able to transfer files directly from Teams (SharePoint Azure Blobs) to the supercomputer. This would be much better in every way.
  - Send as single archive?
  - Send in multiple batches so that processing can start even as more files are arriving?
  - How do you move files from the scratch space to the local storage on the node?
* Could also send files back right to SharePoint?

## Documentation
### README.md
* FileZilla information should be updated when we figure out how to integrate SharePoint
* CLI commands and flags will need to change once we update buzzdetect.py

# So much done!
## Preprocessing pipeline
* Do we need to get Reed's snipper code working?
  - New chunking code working that splits audio into 1h segments, but doesn't downsample or snip out buzzes only; I don't think we have a need to snip out buzzes at this point
*  MP3 → WAV; on supercomputer instead of local?
  -  No longer converting to WAV, but chunking on supercomputer probably makes sense; parallelizing would be great!

## Code
### train.py
* [done!] Store metadata csv within model directory

### analyze.py
* Output confidence score as probability?
    - Apparently, working with raw confidence scores is normal and acceptable; instead of probability, I need to figure out a confidence cutoff that balances false positives/negatives
 
## Machine Learning Design
* Make sure you're doing biasing correctly! Does the sample from the training set apply to the sample from the field audio?
    - I'm going to call this one done just because I feel comfortable with the biasing at the moment; maybe the biases should shift the prior expectation from rates-in-training to rates-in-field, but rates-in-field can't really be known ahead of time.
 
## Supercomputing
### Figure out Teams → OSC file transfer
* Are we working with terabytes at once?
  - No. Lily's entire experiment is only 67GB (assuming it's all uploaded on Teams)
 
## Documentation
### README.md
* FileZilla information should be updated when we figure out how to integrate SharePoint
    - Bummer...FileZilla doesn't work with SharePoint unless you pay for Pro. Which honestly might be worth it...
