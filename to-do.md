# So much to do...
## Before v1.0
### Essential
* Rewrite CLI options and test all actions
* Write CLI help
* Rewrite README documentation
* Purge buzzes from ambient_day files
* Add sufficient ambient_night annotations
* Migrate classification dictionary into a .md in git repository
* Re-name labels to fit with class dictionary 2.0
     - For classes that were split, just leave the labels that haven't been reclassified at the finest known classification. E.g., "mech_auto_RECLASSIFY" can become mech_auto
     - RECLASSIFY is a mix of human and scrapes; mostly human. Just call human for now?
* Train new models on relabeled data
  * Model 1: full data, weight of 1 for each class
  * Model 2: full data, weight of (total training duration/selected duration)

### Optional
* Write docstrings for functions
* Write custom loss function

## Code
### Overall
* Add docstrings to functions

### train.py
* Store training history as python object in model directory (see the pickle package for python)
* Allow option to run test_model.py after building model
     - Will I get tensorflow problems when multithreading?
     - Isn't this already being done in training? Can I just save those results?
- YAMNet doesn't ignore incomplete frames. Do I need to prune them before training? Maybe it doesn't makea  big difference? But it could. If you had a class with all 1.0s audio, you would train it on 50% empty frames

### analyze.py
* Allow cacheing of YAMNet embeddings for repeat analysis
* Add progress bar/estimated time to completion (dependent on verbosity?)
* Split analyze_data() into extract_embeddings() and analyze_embeddings()
* Use full paths (not clipped) at verbosity 2

## Machine Learning Design
* Make ambient_night classification
* Listen to all ambient_day and remove buzzes
* Write custom loss function that penalizes within-buzz error less
* Add dense layer for hierarchical categorization (categorize buzz, then within buzz categorize insect)
* Move away from YAMNet embeddings? 

## Supercomputing
### Figure out supercomputing
* Run on GPU

## Documentation
### README.md
* CLI commands and flags will need to change once we update buzzdetect.py
