Setting up buzzdetect for the first time
==========================================

1. Get conda
------------

Install the package manager `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. Any version of conda should work for building the environment, but we've been using version 23.7.3.

2. Get files
------------

Clone the buzzdetect files from this github repo to the directory where you want to store buzzdetect. We'll refer to this directory as the "project directory."

3. Build environment
--------------------

The project directory contains a file named `environment.yml <https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/environment.yml>`_. This file specifies all of buzzdetect's dependencies. Using conda, we can install all of the packages buzzdetect requires into one location.

You can install your conda environment wherever you wish, but in this guide, we will install it in the same directory as the buzzdetect project.

1. Open a terminal in the same directory as your YAML file
2. Run the command: ``conda env create -f environment.yml -p ./environment``

   * the -p argument is the path where the conda environment will be created. So long as the environment is accessible to all users, location doesn't matter.
   * if you chose to build the environment elsewhere or in the default location with a name, replace the ``./environment`` argument in future commands with your chosen path or name.

3. Conda will create the ``./environment`` subdirectory and install an environment with all required packages there.

4. Confirm proper installation
-------------------------------

Despite specifying the environment name as the first line of the YAML file, conda doesn't appear to associate the name with the environment when using the -p option. Because of this, we'll have to activate it with the file path.

1. Activate the conda environment by running ``conda activate ./environment``.
2. Run ``conda list``; you should be able to see all of the specified dependencies

5. Test an audio file
---------------------

buzzdetect comes prepackaged with our latest model (the name will change, but look in ``./models``) and an audio analysis file, ``testbuzz.mp3``.
Let's try running our first analysis.

1. Find the name of the model you want to try out. The name is the directory name under ./models. For example, ./models/model_general_v3 represents the model named "model_general_v3".
2. Open a terminal in the project directory
3. Run the command ``conda activate ./environment``
4. Run the analysis command, inserting the model name of whichever model you want to run.
   E.g.: ``python buzzdetect.py analyze --modelname model_general_v3``

You should see the terminal print out information about the analysis; one analyzer will launch and process the file. The results and log file will be written to an 'output' subdirectory within the model's directory.