# Analyzing audio with buzzdetect

## Inputs: audio files
buzzdetect is capable of analyzing any audio filetype [supported by the soundfile package](http://www.mega-nerd.com/libsndfile/#Features). The files are decoded to raw data in-memory and resampled to the appropriate samplerate. [YAMNet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet.py), the model we're using to extract embeddings, expects an input of 0.960 seconds of audio at 16,000Hz, so input audio is framed in frames of length 0.960s for analysis.

### File size and memory useage
Inputs are loaded in chunks and so can be arbitrarily large. You can control chunk size (and thus memory used) with the `--memory` argument in command line use of [buzzdetect.py](https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/buzzdetect.py). Leaving the argument empty defaults to a memory allotment of 1GB/process. Memory useage is only an estimation, but it errs conservatively. Memory allotment higher than 2GB/process provides marginal performance gains and increases the time between file writes meaning more work is lost if the analysis is interrupted. Increasing memory allotment from 0.5GB/process to 1GB/process increases total analysis rate by ~9%; from 1GB/process to 2GB/process increases rate by ~2.3%; 2GB/process to 3GB/process increases rate by ~1.7%

## Outputs: _buzzdetect.csv files
Results are output as CSV files. Every row of the CSV is one frame of audio (0.960s). An output CSV has four columns:
  - start: the timestamp of the audio file when the frame begins
  - end: the timestamp of the audio file when the frame ends
  - class_prediced: the label corresponding to the model's class prediction
  - score_predicted: the activation of the neuron on the output layer corresponding to the predicted class. You can interpret this as a metric of confidence in the prediction, with high values representing high confidence and low values representing uncertainty. Converting the neuron activation to a probability is fraught, so we have left the value unmodified. It may be possible to create more conservative predictions by filtering the output data for frames where score_predicted is above some threshold.

Output files are named identically to the input audio file with the extension removed and tagged with "_buzzdetect.csv". Because results are specific to the model used in analysis, output files are stored in the `/output` subdirectory of the model used in analysis. This prevents accidental mixing of models used in an analysis and facilitates rapid testing of many different models. Directory structure in the input directory (`./audio_in`, by default) is cloned to the output directory. Thus, if the input files have the structure of:


```
buzzdetect/
└── audio_in/
    └── experiment_foraging/
        ├── recorder_1/
        │   ├── 2023-02-01.mp3
        │   └── 2023-02-10.ogg
        ├── recorder_2/
        │   ├── 2023-02-01.flac
        │   └── 2023-02-09.wav
        └── backyard_test.mp3
```

and they are analyzed with the model "agricultural_01", they will be output as:

```
buzzdetect/
└── models/
    └── agricultural_01/
        └── output/
            └── experiment_foraging/
                ├── recorder_1/
                │   ├── 2023-02-01_buzzdetect.csv
                │   └── 2023-02-10_buzzdetect.csv
                ├── recorder_2/
                │   ├── 2023-02-01_buzzdetect.csv
                │   └── 2023-02-09_buzzdetect.csv
                └── backyard_test_buzzdetect.csv
```



## Parallel processing
buzzdetect can analyze a single file with multiple processes, so feed it as many cores as you have! Total performance peaks when the number of processes is equal to the number of logical CPUs on your machine. Results are written as they are produced, so if analysis of a large file is interrupted part-way through, you shouldn't lose much progress. Just run the analysis again and buzzdetect will pick up where it left off.

## Transfer learning
buzzdetect employs transfer learning for audio classification. The models stored in the `./models` directory do not directly analyze audio data but instead analyze the embeddings created by YAMNet. Analysis of a file follows these steps:
1. A chunk of audio data is read into memory
2. The audio data is resampled to 16KHz and framed into 0.960 second chunks
3. YAMNet is applied to each frame, producing embeddings for each frame
4. Our bespoke models are applied _to the YAMNet embeddings_, producing predictions for each frame
5. Predictions are appended to the corresponding CSV file

