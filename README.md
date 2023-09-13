# BuzzDetect

## Get started by accessing the terminal
1. Log in

    i. Visit www.osc.edu and click “login” in the top-right corner

   ii. Click “OSC OnDemand”
   
   iii. Enter your username and password

2. Navigate to the project directory

   i. Click files
   
   ii. Navigate to /fs/ess/PAS1063/buzzdetect
   
   iii. Within the buzzdetect folder, click “open in terminal”

3. There is a conda environment that has all necessary modules installed and configurations made. Load the environment by typing `conda activate ./environment`

   - If you get the error `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`, run `conda init bash` and then rerun the activate command

## To analyze audio in bulk (several audio files)

1. Follow the instructions under “Get started by accessing the terminal” (step 3 is important)
2. Create a folder containing the audio files you want. The folder should not contain any additional files. It’s recommended you put the folder inside of the loadingdock folder.
3. Complete any preprocessing

    a. Files must be .wav files
   
    b. Use `python buzzdetect.py –action preprocess --preprocesspath [path to directory]` to ensure each .wav file is 16bit
   
   Note: preprocesspath demands a directory containing the audio files to preprocess. If you are looking to preprocess a single audio file, put it in an otherwise empty directory and use that directory’s path
4. Run the buzzdetect.py script, including the following flags:

    `--action analyze`
   
    `--type bulk`
   
    `--analyzesrc [path to directory]`

    - See the comprehensive command-line documentation [here](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation_CLI.md) for additional configuration options.

    - The whole command will look something like this: `python buzzdetect.py --action analyze --type bulk --analyzesrc loadingdock/togo`
   
    - By default .txt files will be output into the bulkoutput folder. If the file already exists, it will be appended to (not overwritten), which may cause problems

    - Audio files will not be deleted.

## To analyze a single audio file

Follow the same instructions as above, but for step 3b, use --type bulk, and input the path to the file for --analyzesrc

## Retraining the model

## Uploading files with FileZilla
Files can be uploaded with drag-and-drop, or with the “Upload” button, however, when you need to upload a lot of data (many files or few, large files), FileZilla makes things easier.

1. Download/install at https://filezilla-project.org/download.php
2. Click the “Site Manager” button in the top left:

    <img src="filezilla_sitemanager.png" width="400">=
3. The lefthand file selector is for files on your computer; the righthand file selector contains files on the server
4. Navigate to /users/PAS1063/jcunderwood/bee-audio on the righthand pane and drag the files you want to move from your computer

## Complete buzzdetect.py command-line documentation
Complete buzzdetect.py command-line documentation can be found [here](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/documentation_CLI.md). (still a wip) 

## Misc.
The legacy folder is a sort of graveyard of old scripts, as well as testing data, that are not used by BuzzDetect. If you’re trying to run some kind of analysis/testing/processing that is not supported off-the-shelf, you may find a script here that does what you want.


— James Underwood / jcunderwood@protonmail.com
