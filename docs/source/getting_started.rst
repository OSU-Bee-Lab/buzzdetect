Setting up buzzdetect for the first time
==========================================

1. Get Conda
--------------
Install the package manager `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

2. Get files
--------------
1. Open a terminal wherever you want to store the buzzdetect folder.
2. git clone the buzzdetect files from `the GitHub repo <https://github.com/OSU-Bee-Lab/buzzdetect>`_.

    - Command: ``git clone "https://github.com/OSU-Bee-Lab/buzzdetect" buzzdetect``
    - This installs buzzdetect to the subdirectory "buzzdetect" under your current working directory.
    - We'll refer to the buzzdetect folder in your chosen location as the "project directory."

.. _conda_env:
3. Install dependencies
------------------------
The project directory contains a file named `environment.yml <https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/environment.yml>`_.
This file specifies all of buzzdetect's dependencies.
Using Conda, we can install all of the packages buzzdetect requires into one location.

**NOTE:** if you want to use a GPU to (drastically) speed up processing, you must modify environment.yml.
Change the ``- tensorflow`` section to read ``- tensorflow[and-cuda]``.
E.g., ``- tensorflow>=2.16`` should become ``- tensorflow[and-cuda]>=2.16``.
If this is not done, all processing will occur on CPU.

You can install your Conda environment wherever you wish, but these steps will install it in the same directory as the buzzdetect project.

1. Open a terminal in the the project directory
2. Run the command: ``conda env create -f environment.yml -p ./.venv``

    - The -p argument is the path where the Conda environment will be created

3. Conda will create the ``./.venv`` subdirectory and install a virtual environment with all required packages there.

4. Confirm proper installation
-------------------------------
1. Activate the conda environment by running ``conda activate ./venv``.
2. Run ``conda list``; you should be able to see all of the specified dependencies (along with many other packages).

5. Test an audio file
---------------------

buzzdetect comes prepackaged with our latest model (the name will change, but look in `the models directory <https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/models>`_.)
and an audio analysis file, `testbuzz.mp3 <https://github.com/OSU-Bee-Lab/buzzdetect/blob/main/audio_in/testbuzz.mp3>`_.
Try running an analysis!

    - If you like buttons, read :doc:`the GUI documentation <gui>`.
    - If you're savvy in the command line, read :doc:`the command line documentation <cli>`.
    - If you're using buzzdetect in another Python project, check out :doc:`the API documentation <api>`.

For further reading, see :doc:`workflow`.

Happy listening!