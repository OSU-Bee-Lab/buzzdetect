# Setting up buzzdetect for the first time
#### conda version
The version of conda available on OSC is 4.12.0. For some reason related to privileges, I was not able update conda to the current version of 23.7.4. Nonetheless, I never ran into any problems

## 1. Setup
Clone the buzzdetect files from this github repo to the directory where you want to store buzzdetect. We'll refer to this directory as the "project directory."

## 2. Create the YAML (.yml) file to specify packages
If you cloned the github repo, you will already have a file named "environment.yml" in the project directory (a so-called YAML file). If you don't have the file, the contents of the file can be seen [here](https://github.com/OSU-Bee-Lab/BuzzDetect/blob/main/environment.yml).

## 3. Build from the YAML file
You can install your conda environment wherever you wish, but in this guide, we will install it in the same directory as the buzzdetect project.
1. Open a terminal in the same directory as your YAML file
2. Run the command: `conda env create -f environment.yml -p ./environment`
  - the -p argument is the path where the conda environment will be created. I've chosen to create it as a subdirectory of the project directory. So long as the environment is accessible to all users, location doesn't matter.
3. Conda will create the `environment` subdirectory and install an environment with all required packages there.


## 4. Confirm proper installation
Despite specifying the environment name in the YAML file, conda doesn't appear to associate the name with the environment when using the -p option. Because of this, we'll have to activate it with the file path.
1. Activate the conda environment by running `conda activate ./environment`.
2. Run `conda list`; you should be able to see all of the specified dependencies
