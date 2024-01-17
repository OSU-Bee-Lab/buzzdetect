# So much to do...
## Code
### Overall
* Adjust useage of chunklength to always be in seconds once chunklength is automatically set from memory and cores
* Add docstrings to functions

### train.py
* Store training history as python object in model directory (see the pickle package for python)
* Allow option to run test_model.py after building model
     - Will I get tensorflow problems when multithreading?
     - Isn't this already being done in training? Can I just save those results?

### analyze.py
* Allow cacheing of YAMNet embeddings for repeat analysis

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
