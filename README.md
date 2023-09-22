# BuzzDetect

## Getting set up on OSC
1. Log in to OSC

    i. Visit www.osc.edu, click the "Access OSC" dropdown in the top right, and click System Gateway
    <img src="/documentation/OSC_login.png" width="600">
   
   ii. Log in with your username and password

2. Navigate to the project directory

   i. Click the "Files" dropdown in the top left

   ii. Click "/fs/ess/PAS1063"
   
   You are now in the "project directory" for the Bee Lab. Files made here will, by default, be accessible to anyone else on this project. This differs from your home directory, which has a path of /fs/users/PAS1063/[your username]. Note: if you were first granted access to the supercomputer on another project, your home directory won't have PAS1063, but the ID of the first project you were granted access to.
   
   ii. Navigate to /fs/ess/PAS1063/buzzdetect
   
   iii. Within the buzzdetect folder, click “open in terminal”

4. Activate the conda environment
    - The conda environment is a location where all packages for the buzzdetect tool are pre-configured
    - If there is a directory within buzzdetect/ called "environment", the conda environment has already been set up. Load the environment by typing `conda activate buzzdetect-py39`
        - If you get the error `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`, run `conda init bash` and then rerun the `activate` command
    - If there is no "environment" directory, see [the time first setup documentation](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation/documentation_firstTimeSetup.md), then return to this step.
  

## Getting set up locally
1. Ensure all your files are up-to-date with the GitHub repo
2. Install python and conda on your device [need more detail!]
3. Go through the [time first setup process](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation/documentation_firstTimeSetup.md)
4. You're ready to go! Load or train a model and get detecting.

## To analyze audio
What follows is a basic workflow for the buzzdetect tool. If you want refined control over the analysis, see the [command line documentation](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation/documentation_CLI.md).
1. Place the files you wish to analyze in ./buzzdetect/audio_in
       - At the time of writing, files need to be WAV files with 16 bit depth and cannot be hours long (exact filesize limit unknown)
       - At the time of writing, buzzdetect's preprocessing function is not functioning
2. Have a trained model ready in the ./buzzdetect/models/ directory
    - See the section "To train a model" below
4. Open a terminal in the project root directory (./buzzdetect/) and activate the conda environemnt with `conda activate ./environment`
5. Analyze the audio files with the following command: `python buzzdetect.py analyze --modelname [your model's name here]`
    - See the comprehensive command-line documentation [here](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation/documentation_CLI.md) for additional configuration options.
6. The analysis will be output as .txt files in the directory ./buzzdetect/output
    - Note: running the analysis multiple times will append (not overwrite) the new information to the old file; this may cause problems

## To train a model
New models can be trained with the command `python buzzdetect.py train --modelname [your model name here] --trainingset [your training set here]`
"Training set" refers to a metadata CSV, stored in ./buzzdetect/training/ with the name of "metadata_[setname].csv". The trainingset option looks for a metadata CSV with a matching name, subsets the training data (in the ./buzzdetect/training/audio directory) for matching audio files, and trains a model on them.

The model is output as a folder to ./buzzdetect/models. The folder is then read by the analyze function of buzzdetect.py for analysis.

## Uploading files with FileZilla
Files can be uploaded with drag-and-drop, or with the “Upload” button, however, when you need to upload a lot of data (many files or few, large files), FileZilla makes things easier.

1. Download/install at https://filezilla-project.org/download.php
2. Click the “Site Manager” button in the top left:

    <img src="/documentation/filezilla_sitemanager.png" width="400">
3. The lefthand file selector is for files on your computer; the righthand file selector contains files on the server
4. Navigate to /fs/ess/PAS1063/buzzdetect on the righthand pane and drag the files you want to move from your computer
