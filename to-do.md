# So much to do...
## Documentation
* Write docstrings for functions and modules
* Make guide for managing training data and training new models. Make way for other researchers to share training data.

## Code
### training
* Allow training from cached embeddings. The issue here is that I can't write custom python functions to load embeddings. I can't pull strings out of tensors to operate on filepaths and as best as I can find, there's no tensorflow function for loading arbitrary data.
* Store training history as python object in model directory (see the pickle package for python)
* Allow option to run test_model.py after building model
     - Will I get tensorflow problems when multithreading?
     - Isn't this already being done in training? Can I just save those results?

### analysis
* Given that I'm not padding, what happens between the end of one chunk and the start of the next? If the chunk isn't defined in a number of frames, am I leaving tiny gaps at the end of each chunk?
       - More importantly, should I redefine chunks to be a number of frames? A number of audio samples (probs better for memory estimation)?
* Add progress bar/estimated time to completion (dependent on verbosity?)
* Use full paths (not clipped) at verbosity 2
* Try stretching audio; YAMNet expects (0.960*16,000) samples. Can we instead feed it 0.348s of 44.1KHz audio? BirdNET expects 3s of 48KHz audio...probably nothing we can do to satisfy that with a smaller frame length.
* Parallel processing: can I account for the edge case where chunk size is _too_ large? E.g., you have 8 cores, each taking chunks of 3600 seconds, but only a single file that's 3600 seconds long. Optimal would be reducing chunk size.
* Memory allotment: if reading in files of higher samplerate, memory useage will exceed what's anticipated until resampling is completed.

## Machine Learning Design
* Add dense layer for hierarchical categorization (categorize buzz, then within buzz categorize insect)
* Move away from YAMNet embeddings?
* Try random forest
