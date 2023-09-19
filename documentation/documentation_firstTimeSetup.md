# Setting up buzzdetect for the first time

## Directory structure
Directory structure is still very much in flux; I'm sure as soon as I'm done I'll come right back to the document and update it. There's no chance some future researcher or grad student will come across this "under construction" note, because I'm not the kind of guy that leaves documentation half-finished. No siree.

## Create the cona environment
### conda version
This documentation was written in regards to conda version 23.7.4. The version available on OSC was 4.12.0. I never encountered problems that were definitely the result of an out-of-date conda version, but I brought my conda up to date as a precaution. To update conda to the latest version, run `conda update -n base -c defaults conda`.

### Create the YAML (.yml) file to specify packages
Put the following text in a text file with the extension `.yml` (a so-called YAML file):
  
```
name: buzzdetect-py39
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.9
  - pip=23.2.1
  - pip:
    - tensorflow==2.13
    - pydub==0.25.1
    - tensorflow_io==0.34.0
    - tensorflow_hub==0.14.0
    - ffmpeg==1.4
```

Note: despite specifying the environment name in the YAML file, conda doesn't appear to pick up on the name when it's installed in the project directory instead of your home directory. Thus, we'll have to activate it with the file path.

Save the YAML file to your directory of choice. The location doesn't matter, but it makes the most sense to keep it in the project directory.

### Build from the YAML file
1. Open a terminal in the same directory as your YAML file
2. Run the command: `conda env create -f environment.yml -p /fs/ess/PAS1063/buzzdetect/environment`
  - the -p argument is the path where the conda environment will be created. At the time of writing (September 2023), it was `/fs/ess/PAS1063/buzzdetect`, but change as needed.
3. Conda will create the `environment` subdirectory and install an environment with all required packages there.

### Confirm the environment works
Activate the conda environment with `conda activate /fs/ess/PAS1063/buzzdetect/environment` (or, if you're already in the buzzdetect directory, just: `conda activate ./environment`. If everything installed correctly, you should receive no errors and should be able to see all of the specified dependencies when you run `conda list`.
