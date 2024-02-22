# Training data and associated files
## The ./training directory
### /audio
The directory holding audio files for training purposes. Audio files are snipped from the raw data using annotations. The raw data files are MP3s with a samplerate of 44.1KHz.
We write the snips as WAVs to avoid data loss through repeated decoding and encoding.
Though YAMNet expects audio at 16Khz, we preserve the original samplerate of the audio for flexibility and resample in-memory.

### /metadata
The metadata files are necessary to associate training information with the training audio.
We have opted not to use file or directory names to store training information such as classification. This allows us to treat the same data differently with different models.
For example, the raw metadata can be modified so that the `ambient_day` and `ambient_night` classifications are both grouped into the single category `ambient`.
Then, a new model can be trained on the join ambient class without any modification to the training files themselves.

When a model is trained, it stores a copy of its training metadata in its directory.

#### data dictionary for metadata.csv
Every row in a metadata file represents one human-annotated audio file.
Annotations should be "tight" around the event of interest: when manually labeling, if a bee buzzed for 2s, fell silent for 1s, then buzzed again for 1s, each buzz event should be given its own label.
This criterion was not in place for the entirety of our manual labeling, but does seem to hold true for the majority of training data.

Importantly, audio files differ in their length. Thus, one row does **not** represent one training sample. YAMNet expects 0.960s of audio at 16,00KHz and we use a frame hop of 0.5*framelength and no padding.
Thus, 2 seconds of audio yields three training frames: (0, 0.960), (0.480, 1.440), (0.960, 1.920).

A metadata.csv file contains the following columns:
* **ident**

  The ident column serves as a unique identifier of the raw data file from which the training data was snipped.
Essentially, the ident is the relative file path of the raw audio file with the file extension removed.
For ease of manipulating data, we've chosen to preserve the structure of our raw audio files in the training data. This structure is: `experimentName/siteName/recorderID/filename`.
The file name is written in the format `%y%m%d_%H%M`. Thus 220713_0943 is a file recorded on 2022-07-13T09:43.
This also preserves information about correlation between training files. Training files from the same site will have non-independent training data for some sound classifications (e.g., the same jet plane would be picked up by all microphones).

* **classification**

  The training label applied to the file. See [the classification dictionary](https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/documentation/classificationDictionary.md) for more information.

* **duration**

  The duration of the audio file in seconds.

* **path_audio**

  The relative path to the audio file with the root directory as the project directory.

* **fold**

  The training fold the file belongs to. 1–3 is the training set, 4 is the validation fold, 5 is the test fold (never exposed to the model).

* **frames**
  The number of frames that YAMNet will generate from the audio file. Used in weighting to adjust for imbalanced dataset.

  
