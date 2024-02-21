# buzzdetect

## Introduction
buzzdetect is a machine learning based python tool for automated biacoustic surveys of honey bees. buzzdetect is capable of processing arbitrarily large audio files of virtually any type and producing second-by-second annotations of the sounds identified in the files. It outputs results as CSV files for easy transfer into other data processing software.

### Performance
The accuracy of buzzdetect varies between models. Our current best model has a true positive rate for honey bees of ~87% within the test set of the training data, but we are still addressing issues with false-positives when the model encounters sounds outside of its training set.

The speed of buzzdetect varies between models and the machine running the computation. Per-process performance is roughly 20x–40x realtime. An average laptop with a solid state hard drive processing on 8 logical CPUs will probably process an hour of audio in about 15 seconds.

### Models
The models that buzzdetect applies to data are still rapidly evolving. We will publicly host a selection of models that we believe to be production-ready, as well as performance metrics to aid in the interpretation of their results.

### Transfer learning
buzzdetect is based on the machine learning strategy of transfer learning. In brief, this means we do not feed our bespoke models audio data directly, but instead preprocess the data using an existing audio classification model. This allows us to create a highly performant model on a relatively small dataset. The first model extracts relevant features from the audio data and represents them in a lower-dimensional space called the "embedding space." Our models are trained on those smaller and more information-dense embeddings.

Currently, we are training our models on embeddings from the Google Tensorflow model [YAMNet](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet.py).
  

## Quick Start Guide
### First time steup
1. Install the environment manager [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Clone the buzzdetect files from this github repo to the directory where you want to store buzzdetect. We'll refer to this directory as the "project directory."
3. Open a terminal in the project directory
4. Run the command: `conda env create -f environment.yml -p ./environment`
  - the -p argument is the path where the conda environment will be created. The above command creates the environment as a subdirectory of the project directory, but you can choose to install it elsewhere.
5. Conda will create the environment and install all required packages

You are now ready to run your first analysis!

### Analyzing audio
1. buzzdetect looks for audio files to analyze in the `./audio` subdirectory. Place the audio you want to analyze in this directory.
       - Audio files can be of any format that [the soundfile package](https://python-soundfile.readthedocs.io) supports; see supported types [here](http://www.mega-nerd.com/libsndfile/#Features).
       - Audio files do not need to be in the root level of the `./audio` subdirectory. You can have them stored in any structure you like. buzzdetect will clone the input directory structure to the output folder. For example, the results of the audio file `./audio_in/visitation_experiment/block 1/recorder4.wav` would be written to `./models/[model used in analysis]/output/visitation_experiment/block 1/recorder4_buzzdetect.csv`
2. Open a terminal in the project directory and activate the conda environment by running the command `conda activate ./environment`
5. Analyze the audio files with the following command: `python buzzdetect.py analyze --modelname [model to use] --cpus [number of cpus]`
    - See the comprehensive command-line documentation [here](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation/documentation_CLI.md) for additional configuration options.
6. The results will be output as .csv files in the `./output` subdirectory of the model's directory.
