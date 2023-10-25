# So much to do...
## Preprocessing pipeline

## Code
### Multithreading
* Well shoot, it looks like tensorflow is already multithreaded? When runing an analysis, on the ML step all cores are equally loaded (though only to ~50%) on my machine and there are 10 processes running with the command `python buzzdetect.py analyze --modelname [model]`. When a chunking step is running, one core is utilized with the ffmpeg command. So maybe the approach is to multithread-chunk files up to a storage limit, then run ML one chunk at a time. I think there are significant performance gains to be had here, because the ffmpeg step does take a long time to run with 1h chunks.
   - This kinda invaldiates the multithreading script I started last night (though that would still work for a distributed compute, just saying).

### train.py
* Store training history as python object in model directory (see the pickle package for python)
* Allow to read from different directories

### analyze.py
* Throw out last frame that overruns audio file
* Detect whether output file conflicts ahead of time instead of just appending (maybe just if the dir already exists, make a new dir with a timestamp)
* Output analysis metadata (what model was used? When was the analysis run? Framerate? Framehop? Other things?)
* Add verbosity (change ffmpeg default to "converting [chunk]", add "analyzing [chunk]")
* Check memory usage [(this stack overflow article proposes a good tool)](https://stackoverflow.com/questions/131303/how-can-i-measure-the-actual-memory-usage-of-an-application-or-process)

### To build
* MAKE script to create directory structure

## Machine Learning Design
* Make "none"/"ambient" audio in training set

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
