# BuzzDetect

## Get started by accessing the terminal
Log in
Visit www.osc.edu and click “login” in the top-right corner
Click “OSC OnDemand”
Enter your username and password
Click files
Navigate to /users/PAS1063/jcunderwood/bee-audio
Within the bee-audio folder, click “open in terminal”
Load python 3.9 with Anaconda:
Type module load python/3.9-2022.05
Type source activate local

## To analyze audio in bulk (several audio files)

Follow the instructions under “Get started by accessing the terminal” (step 3 is important)
Create a folder containing the audio files you want. The folder should not contain any additional files. It’s recommended you put the folder inside of the loadingdock folder.
Complete any preprocessing
Files must be .wav files
Use python buzzdetect.py –action preprocess --preprocesspath [path to directory] to ensure each .wav file is 16bit
Note: preprocesspath demands a directory containing the audio files to preprocess. If you are looking to preprocess a single audio file, put it in an otherwise empty directory and use that directory’s path
Run the buzzdetect.py script, including the following flags:
--action analyze
--type bulk
--analyzesrc [path to directory]
See the comprehensive command-line documentation here for additional configuration options
python buzzdetect.py --action analyze --type bulk --analyzesrc loadingdock/togo
By default .txt files will be outputted into the bulkoutput folder. If the file already exists, it will be appended to (not overwritten), which may cause problems
Audio files will not be deleted.

## To analyze a single audio file

Follow the same instructions as above, but for step 3b, use --type bulk, and input the path to the file for --analyzesrc

## Retraining the model

## To upload large/many files
Files can be uploaded with drag-and-drop, or with the “Upload” button.
However, when you need to upload a lot of data, you can use FileZilla
Download/install at https://filezilla-project.org/download.php
Click the “Site Manager” button in the top left:
<img src="filezilla_sitemanager.png" width="400">

The lefthand file selector is for files on your computer; the righthand file selector contains files on the server
Navigate to /users/PAS1063/jcunderwood/bee-audio on the righthand pane and drag the files you want to move from your computer

## Complete buzzdetect.py command-line documentation
Complete buzzdetect.py command-line documentation can be found [here](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation_CLI.md). (still a wip) 

## Misc.
The legacy folder is a sort of graveyard of old scripts, as well as testing data, that are not used by BuzzDetect. If you’re trying to run some kind of analysis/testing/processing that is not supported off-the-shelf, you may find a script here that does what you want.


— James Underwood / jcunderwood@protonmail.com
