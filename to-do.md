# So much to do...
## Documentation
* Write docstrings for functions and modules
* Make guide for managing training data and training new models. Make way for other researchers to share training data.

## Code
### general
* Change framelength to be number of samples, not time (makes some operations like)
* The warmup for analysis is taking forever lately. We're running it on all data files the lab has---perhaps the merge_chunks() function has some inefficiency that's bogging it down when there are many files? Profile the code.

### training
* Restore loss from when patience triggered, not best weights
* Implement data augmentation
       - Overlapping buzzes with ambient and other sounds
       - Add gaussian noise to buzz? Might be more generalized than agricultural ambient noise
       - Vary amplitude of buzzes
       - Smaller framehop for buzzes?

### analysis
* Drop "end" and "class_max" columns; they aren't relevant to current analysis pipelines
* Add pre-embedding of entire audio files
  - Shouldn't need too much RAM allotment. 260,000 frames ~= 1,080MB. 1h of audio at 0.5 framehop = 7500 frames. Should be under 32MB data.
* Redefine chunks to be a number of audio samples (for memory estimation)
* Work in frame numbers instead of timestamps? Where frame 1, 2, 3, 4 is ~2.5s of audio because of overlap
     - Timestamps are getting messy in the CSV (e.g., '4923.999999999') and I think this may cause problems when picking an analysis back up
     - Would also facilitate pre-embedding long files
* Use full paths (not clipped) at verbosity 2
* Parallel processing: can I account for the edge case where chunk size is _too_ large? E.g., you have 8 cores, each taking chunks of 3600 seconds, but only a single file that's 3600 seconds long. Optimal would be reducing chunk size.
* Memory allotment: if reading in files of higher samplerate, memory useage will exceed what's anticipated until resampling is completed.

### QOL
#### Analysis
* Add progress bar/estimated time to completion (dependent on verbosity?)

## Machine Learning Design
### YAMNet
* Pare down YAMNet design; we've been using a pretty pre-packaged solution that does some internal framing itself; there's likely overhead since I'm pre-framing.
Check out [the github](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/inference.py) and see if we can use the raw model.
### New directions
* Random forest model
* [Phan et al., 2019](https://arxiv.org/abs/1811.01092) have a wicked cool model...arguably more suited to our kind of analysis.

## Moonshots and big lifts
* Rewrite training to use audio at native samplerate; try 44.1KHz audio instead of 16KHz and just don't worry that it stretches out the sound
* Retrain final layers of YAMNet during model training
