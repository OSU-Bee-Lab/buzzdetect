# Setting up buzzdetect for the first time
  ## Create the cona environment
  YAML file contents:
  
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
```
  cd into the project directory and run: `conda env create -f environment.yml -p /fs/ess/PAS1063/buzzdetect/environment`; the -p argument is the path where the conda environment will be created. At the time of writing (September 2023), it was `/fs/ess/PAS1063/buzzdetect`, but update as needed.
