# So much to do...
## Documentation
* Write docstrings for functions and modules
* Make guide for managing training data and training new models. Make way for other researchers to share training data.

## Code
### training
* Store training history as python object in model directory (see the pickle package for python)
* Allow option to run test_model.py after building model
     - Will I get tensorflow problems when multithreading?
     - Isn't this already being done in training? Can I just save those results?

### analysis
* Redefine chunks to be a number of frames? A number of audio samples (probs better for memory estimation)?
* Work in frame numbers instead of timestamps? Where frame 1, 2, 3, 4 is ~2.5s of audio because of overlap
     - Timestamps are getting messy in the CSV (e.g., '4923.999999999') and I think this may cause problems when picking an analysis back up
* Add progress bar/estimated time to completion (dependent on verbosity?)
* Use full paths (not clipped) at verbosity 2
* Try stretching audio; YAMNet expects (0.960*16,000) samples. Can we instead feed it 0.348s of 44.1KHz audio? BirdNET expects 3s of 48KHz audio...probably nothing we can do to satisfy that with a smaller frame length.
* Parallel processing: can I account for the edge case where chunk size is _too_ large? E.g., you have 8 cores, each taking chunks of 3600 seconds, but only a single file that's 3600 seconds long. Optimal would be reducing chunk size.
* Memory allotment: if reading in files of higher samplerate, memory useage will exceed what's anticipated until resampling is completed.

## Machine Learning Design
* Add dense layer for hierarchical categorization (categorize buzz, then within buzz categorize insect)
* Move away from YAMNet embeddings? First layer uses YAMNet, second is fully bespoke?
* Try random forest
