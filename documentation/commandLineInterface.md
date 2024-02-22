# Using buzzdetect from the command line

By interfacing with buzzdetect.py, you can apply models to data to generate labeled audio or train new models.

## Analysis
Analyze files using the `analyze` action; e.g., `python buzzdetect.py analyze ...`

See detailed analysis documentation [here](https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/documentation/analysis.md).

### Options
#### modelname (required)
The name of the model to be used for analysis. Must correspond to the name of a directory in `./models`

#### cpus (default: 1)
The number of analysis processes to create. Multiple processes can analyze the same file. Performance peaks when the number of processes is equal to the number of logical cores on your machine, so throw as many cores as you can spare at the analysis.

#### memory (default: 1*cpus)
The memory limit for buzzdetect. This value is an approximation, but a conservative one. The memory allotment controls the size of chunks that buzzdetect loads into memory for processing.
More memory always increases processing speed, but returns diminish harshly after ~1GB/process. Because buzzdetect write results on a per-chunk basis, longer chunklengths risk more data lost if an analysis is interrupted.
If you are running the analysis on a dedicated machine with plenty of memory, use all of it. If you are running the analysis on your personal machine as you do other work, use less.

#### dir_raw (default: ./audio_in)
The directory holding the audio files to be analyzed.

#### dir_out (default: ./models/[modelname]/output)
The directory to which result files should be written.

#### verbosity (default: 1)
The level of detail printed to the terminal during analysis. Events are logged to the log file in full detail.

## Training
Train new models using the `train action`; e.g. `python buzzdetect.py train ...`

See detailed training documentation [here](https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/documentation/training.md)https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/documentation/training.md.

#### modelname (required)
The name of the model to be trained. The model will be saved in the directory: `./models/[modelname]`

#### metadata (default: metadata_raw)
The name of the metadata file to use for training. This must match the name of a file in `./training/metadata`.
You do not need to write the file extension; e.g. "metadata.csv" will be interpreted the same as "metadata".

#### weights (default: None)
The name of the weights file to use for training. This must match the name of a file in `./training/weights`.
You do not need to write the file extension; e.g. "metadata.csv" will be interpreted the same as "metadata".
If you do not specify weights, each class will be weighted by the natural log of the inverse of the proportion of the training volume it represents.
E.g., if the label "goose" has 100 training frames out of a total training set of 3000 frames, it will receive a weight of:
ln(3000/100) = 5.7037

This adjust for imbalances in the training set.

#### epochs (default: 100)
The **maximum** number of epochs that the model should train for.
We use early stopping to avoid overfitting, so this number may not be reached before the model finishes training.
Most of our models train for ~30 epochs before training is finished.
