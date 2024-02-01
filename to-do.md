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

## Machine Learning Design
* Add dense layer for hierarchical categorization (categorize buzz, then within buzz categorize insect)
* Add a single element to YAMNet embedding array that represents fundamental frequency? Could also be added in the second dense layer (or second model) that classifies within-buzz
* Add time of day to model input?
* Write custom loss function that penalizes within-buzz error less
* Move away from YAMNet embeddings? 

## Supercomputing
### Figure out supercomputing
* Run on GPU

## Documentation
### README.md
* CLI commands and flags will need to change once we update buzzdetect.py
