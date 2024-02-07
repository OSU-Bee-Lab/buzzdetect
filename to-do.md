# So much to do...
## Before v1.0
### Essential
* Rewrite CLI options and test all actions
* Write CLI help
* Rewrite README documentation
* Migrate classification dictionary into a .md in git repository
* Re-name labels to fit with class dictionary 2.0
     - For classes that were split, just leave the labels that haven't been reclassified at the finest known classification. E.g., "mech_auto_RECLASSIFY" can become mech_auto
     - RECLASSIFY is a mix of human and scrapes; mostly human. Just call human for now?

### Optional
* Write docstrings for functions
* Write custom loss function

## Code
### Overall
* Add docstrings to functions

### train.py
* Allow training from cached embeddings. The issue here is that I can't write custom python functions to load embeddings. I can't pull strings out of tensors to operate on filepaths and as best as I can find, there's no tensorflow function for loading arbitrary data.
* Store training history as python object in model directory (see the pickle package for python)
* Allow option to run test_model.py after building model
     - Will I get tensorflow problems when multithreading?
     - Isn't this already being done in training? Can I just save those results?

### analyze.py
* Allow cacheing of YAMNet embeddings for repeat analysis
      - This puts me back in the chunking schema; I can't store all embeddings as a single pickle (too large for memory on large audio files), so I would have to chunk. Maybe I can write to a single .txt or .csv using a write manager and read concurrently?
* Add progress bar/estimated time to completion (dependent on verbosity?)
* Split analyze_data() into extract_embeddings() and analyze_embeddings()
* Use full paths (not clipped) at verbosity 2
* With very long output files (e.g., 50 hour file using a model with many classifications and recording all neuron activations), worker_writer seriously struggles with clearing its queue. Perhaps because of thread scheduling, its work piles up until the end of analysis, where it can take many minutes to perform all of the writes (on most recent analysis, 8 minutes to make 64 writes). This is likely because for every chunk it needs to read, append, sort, and write a csv that can be hundreds of megabytes in size. Options for more intelligent writing:
  - Maintain an internal list of pending write operations; perform all write operations for the same file at once
  - [my preference] Write in chunks until the end of analysis; at the end of analysis, stitch together chunks
        - If analysis gets interrupted, re-running analysis should first check for chunked outputs and stitch them together (would need to happen before checking gaps)

## Machine Learning Design
* Add dense layer for hierarchical categorization (categorize buzz, then within buzz categorize insect)
* Enhance embedding array with elements for relevant information:
     - Dominant frequency of frame (could also be used in a second, within-buzz classificatio model)
     - Time of day
* Write custom loss function that penalizes within-buzz error less

## Supercomputing
### Figure out supercomputing
* Run on GPU

## Documentation
### README.md
* CLI commands and flags will need to change once we update buzzdetect.py
